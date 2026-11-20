import asyncio
from contextlib import asynccontextmanager, nullcontext
import json
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
from typing import TypedDict
import unittest
from unittest.mock import patch

from mcp.server.fastmcp import Context, FastMCP
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
import uvicorn

from test_gateway import Encoder
from toolrank_cpu.gateway import Gateway, GatewayConfig, SearchIndex, UpstreamConfig


class Receipt(TypedDict):
    backend: str
    key: str


class Endpoint:
    def __init__(self, name, token, *, stateful=False):
        self.name = name
        self.authorization = f"Bearer {token}"
        self.denied = 0
        self.moved = 0
        self.redirect = False
        self.expire = False
        self.expired_attempts = 0
        self.unavailable = False
        self.calls = []
        self.finish_call = asyncio.Event()
        server = FastMCP(name, stateless_http=not stateful, json_response=not stateful)

        async def replacement(key: str) -> Receipt:
            self.calls.append(key)
            return {"backend": self.name, "key": key}

        @server.tool()
        async def lookup(key: str, ctx: Context) -> Receipt:
            self.calls.append(key)
            if key == "stall":
                await self.finish_call.wait()
            if key == "refresh":
                server.remove_tool("lookup")
                server.add_tool(replacement, name="replacement")
                await ctx.session.send_tool_list_changed()
            return {"backend": self.name, "key": key}

        @server.tool()
        def hidden() -> str:
            raise AssertionError("Discovery-only tool was invoked")

        self.app = server.streamable_http_app()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive)
        if request.headers.get("authorization") != self.authorization:
            self.denied += 1
            await Response(status_code=401)(scope, receive, send)
        elif scope["path"] == "/moved":
            self.moved += 1
            await Response(status_code=500)(scope, receive, send)
        elif self.unavailable:
            await Response(status_code=503)(scope, receive, send)
        elif self.expire and request.method == "POST" and "mcp-session-id" in request.headers:
            self.expire = False
            self.expired_attempts += 1
            await Response(status_code=404)(scope, receive, send)
        elif self.redirect and request.method == "POST":
            body = await request.json()
            if body.get("method") != "tools/call":
                raise AssertionError("Fixture expects a tool call before the next poll")
            self.redirect = False
            self.calls.append(body["params"]["arguments"]["key"])
            await RedirectResponse("/moved", status_code=307)(scope, receive, send)
        else:
            await self.app(scope, receive, send)


@asynccontextmanager
async def serve(endpoint):
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        host, port = bound.getsockname()
        config = uvicorn.Config(endpoint, log_config=None, access_log=False, timeout_graceful_shutdown=2)
        server = uvicorn.Server(config)
        server.capture_signals = nullcontext
        task = asyncio.create_task(server.serve(sockets=[bound]))
        try:
            async with asyncio.timeout(5):
                while not server.started and not task.done():
                    await asyncio.sleep(.01)
            if task.done():
                await task
                raise RuntimeError("Fixture stopped before listening")
            yield f"http://{host}:{port}/mcp"
        finally:
            server.should_exit = True
            await task


class TransportConfigurationTests(unittest.TestCase):
    def test_rejects_insecure_or_ambiguous_endpoints(self):
        config = {"id": "remote", "transport": "streamable_http", "discover": ["lookup"]}
        for url in ("http://example.com/mcp", "ftp://127.0.0.1/mcp", "https://user:pass@example.com/mcp",
                    "https://example.com/mcp#fragment", "/mcp"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                UpstreamConfig.model_validate(config | {"url": url})
        for url in ("https://example.com/mcp?mode=tools", "http://127.0.0.1:8080/mcp", "http://[::1]/mcp"):
            self.assertEqual(UpstreamConfig.model_validate(config | {"url": url}).url, url)
        with self.assertRaises(ValueError):
            UpstreamConfig.model_validate(config | {"url": "https://example.com/mcp", "command": "fixture"})
        with self.assertRaises(ValueError):
            UpstreamConfig.model_validate(config | {"transport": "stdio", "command": "fixture",
                                                     "bearer_token_env": "REMOTE_TOKEN"})


class HTTPGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def add_endpoint(self, name, *, stateful=False):
        variable = f"GATEWAY_HTTP_TEST_TOKEN_{name.upper()}"
        token = f"fixture-{name}"
        self.enterContext(patch.dict("os.environ", {variable: token}))
        endpoint = Endpoint(name, token, stateful=stateful)
        url = await self.enterAsyncContext(serve(endpoint))
        config = {"id": name, "transport": "streamable_http", "url": url, "bearer_token_env": variable,
                  "discover": ["lookup", "hidden", "replacement"], "invoke": ["lookup", "replacement"]}
        return endpoint, config

    async def gateway(self, *configs, refresh=10, timeout=1.5):
        gateway = Gateway(GatewayConfig.model_validate({"servers": configs, "timeout_seconds": timeout,
                                                        "refresh_seconds": refresh}), SearchIndex(Encoder()))
        await self.enterAsyncContext(gateway.running())
        return gateway

    async def catalog(self, gateway):
        found = await gateway.search("Look up records", limit=20)
        return {match["id"]: match for match in found["matches"]}

    async def wait_for_tools(self, gateway, predicate):
        async with asyncio.timeout(6):
            while not predicate(matches := await self.catalog(gateway)):
                await asyncio.sleep(.03)
        return matches

    async def test_mixed_transports_isolate_bearer_tokens_and_duplicate_names(self):
        self.enterContext(patch.dict("os.environ", {"HTTP_PROXY": "http://127.0.0.1:1", "NO_PROXY": ""}))
        first, first_config = await self.add_endpoint("first")
        second, second_config = await self.add_endpoint("second")
        directory = Path(self.enterContext(TemporaryDirectory()))
        fixture = Path(__file__).with_name("mock_server.py")
        stdio = {"id": "local", "command": sys.executable,
                 "args": [str(fixture), "local", str(directory / "state"), str(directory / "calls")],
                 "discover": ["lookup"], "invoke": ["lookup"]}
        gateway = await self.gateway(first_config, second_config, stdio)
        matches = await self.catalog(gateway)
        self.assertEqual(set(matches), {"first/hidden", "first/lookup", "second/hidden", "second/lookup", "local/lookup"})
        for name in ("first", "second", "local"):
            tool_id = f"{name}/lookup"
            result = await gateway.invoke(tool_id, matches[tool_id]["revision"], {"key": "sample"})
            self.assertFalse(result.isError)
            if name == "local":
                self.assertEqual(result.content[0].text, "local:lookup:sample")
            else:
                self.assertEqual(result.structuredContent, {"backend": name, "key": "sample"})
        hidden = await gateway.invoke("second/hidden", matches["second/hidden"]["revision"], {})
        self.assertEqual(hidden.structuredContent["code"], "not_available")
        self.assertEqual((first.calls, second.calls, first.denied, second.denied), (["sample"], ["sample"], 0, 0))

    async def test_stateful_notifications_refresh_the_allowlisted_catalog(self):
        endpoint, config = await self.add_endpoint("remote", stateful=True)
        gateway = await self.gateway(config, refresh=30)
        original = (await self.catalog(gateway))["remote/lookup"]
        response = await gateway.invoke(original["id"], original["revision"], {"key": "refresh"})
        self.assertFalse(response.isError)
        updated = await self.wait_for_tools(gateway, lambda matches: "remote/replacement" in matches)
        self.assertNotIn("remote/lookup", updated)
        replacement = updated["remote/replacement"]
        result = await gateway.invoke(replacement["id"], replacement["revision"], {"key": "fresh"})
        self.assertEqual(result.structuredContent, {"backend": "remote", "key": "fresh"})
        self.assertEqual(endpoint.calls, ["refresh", "fresh"])

    async def test_unavailable_endpoint_is_hidden_then_recovers(self):
        endpoint, config = await self.add_endpoint("remote")
        gateway = await self.gateway(config, refresh=.1)
        match = (await self.catalog(gateway))["remote/lookup"]
        endpoint.unavailable = True
        await self.wait_for_tools(gateway, lambda matches: not matches)
        response = await gateway.invoke(match["id"], match["revision"], {"key": "offline"})
        self.assertEqual(response.structuredContent["code"], "not_available")
        endpoint.unavailable = False
        await self.wait_for_tools(gateway, lambda matches: "remote/lookup" in matches)
        self.assertEqual(endpoint.calls, [])

    async def test_bad_credentials_hide_tools(self):
        endpoint, config = await self.add_endpoint("remote")
        self.enterContext(patch.dict("os.environ", {"GATEWAY_HTTP_TEST_TOKEN_REMOTE": "incorrect"}))
        gateway = await self.gateway(config)
        self.assertEqual(await self.catalog(gateway), {})
        self.assertEqual(gateway.upstreams["remote"].state, "unavailable")
        self.assertEqual(endpoint.denied, 1)
        self.assertEqual(endpoint.calls, [])

    async def test_expired_session_reconnects_without_reissuing_tool_call(self):
        endpoint, config = await self.add_endpoint("remote", stateful=True)
        gateway = await self.gateway(config, refresh=30)
        match = (await self.catalog(gateway))["remote/lookup"]
        endpoint.expire = True
        result = await gateway.invoke(match["id"], match["revision"], {"key": "expired"})
        self.assertTrue(result.isError)
        self.assertEqual(await self.catalog(gateway), {})
        await self.wait_for_tools(gateway, lambda matches: "remote/lookup" in matches)
        self.assertEqual((endpoint.expired_attempts, endpoint.calls), (1, []))

    async def test_same_origin_redirect_never_replays_an_invocation(self):
        endpoint, config = await self.add_endpoint("remote")
        gateway = await self.gateway(config, timeout=.6)
        match = (await self.catalog(gateway))["remote/lookup"]
        endpoint.redirect = True
        result = await gateway.invoke(match["id"], match["revision"], {"key": "one-attempt"})
        self.assertTrue(result.isError)
        self.assertEqual((endpoint.calls, endpoint.moved), (["one-attempt"], 0))

    async def test_timed_out_http_call_is_not_replayed(self):
        endpoint, config = await self.add_endpoint("remote")
        gateway = await self.gateway(config, refresh=.1, timeout=.6)
        match = (await self.catalog(gateway))["remote/lookup"]
        try:
            result = await gateway.invoke(match["id"], match["revision"], {"key": "stall"})
            self.assertEqual(result.structuredContent["code"], "upstream_timeout")
        finally:
            endpoint.finish_call.set()
        await self.wait_for_tools(gateway, lambda matches: "remote/lookup" in matches)
        self.assertEqual(endpoint.calls, ["stall"])
