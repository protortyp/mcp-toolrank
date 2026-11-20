import asyncio
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from test_http_upstreams import Endpoint, serve
from toolrank_cpu.gateway import GatewayConfig, list_upstream_tools


class InventoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_cli_lists_paginated_stdio_and_authenticated_http_without_invocation(self):
        directory = Path(self.enterContext(TemporaryDirectory()))
        endpoint = Endpoint("remote", "fixture-private-auth")
        url = await self.enterAsyncContext(serve(endpoint))
        calls = directory / "calls"
        config = directory / "gateway.json"
        config.write_text(json.dumps({"servers": [
            {"id": "stdio", "command": sys.executable,
             "args": [str(Path(__file__).with_name("mock_server.py").resolve()), "stdio",
                      str(directory / "state"), str(calls)],
             "discover": ["lookup", "removed"], "invoke": []},
            {"id": "http", "transport": "streamable_http", "url": url,
             "bearer_token_env": "TOOLRANK_TEST_INVENTORY_TOKEN", "discover": []},
            {"id": "unavailable", "command": str(directory / "missing executable"), "discover": []},
        ]}))
        cache = directory / "empty-cache"
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "toolrank_cpu.gateway", str(config), "--list-upstream-tools",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=os.environ | {"TOOLRANK_TEST_INVENTORY_TOKEN": "fixture-private-auth", "TOOLRANK_OFFLINE": "1",
                              "XDG_CACHE_HOME": str(cache)},
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        self.assertEqual(process.returncode, 1)
        self.assertEqual(stderr, b"")
        self.assertEqual(json.loads(stdout), {"upstreams": {
            "stdio": {"status": "ready", "tools": ["hidden", "lookup", "unlisted"],
                      "missing_discover": ["removed"]},
            "http": {"status": "ready", "tools": ["hidden", "lookup"], "missing_discover": []},
            "unavailable": {"status": "unavailable", "tools": []},
        }})
        self.assertFalse(cache.exists())
        self.assertFalse(calls.exists())
        self.assertEqual(endpoint.calls, [])
        self.assertEqual(endpoint.denied, 0)
        self.assertNotIn(b"fixture-private-auth", stdout)

    async def test_empty_configuration_returns_an_empty_inventory(self):
        self.assertEqual(await list_upstream_tools(GatewayConfig(servers=[])), {})


    async def test_server_without_tool_capability_is_ready_and_empty(self):
        program = """import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server

async def main():
    server = Server("tool-free")
    async with stdio_server() as streams:
        await server.run(*streams, server.create_initialization_options())

asyncio.run(main())
"""
        config = GatewayConfig.model_validate({"servers": [
            {"id": "resources", "command": sys.executable, "args": ["-c", program], "discover": ["removed"]}
        ]})
        self.assertEqual(await list_upstream_tools(config), {
            "resources": {"status": "ready", "tools": [], "missing_discover": ["removed"]},
        })
