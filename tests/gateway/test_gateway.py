import asyncio
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np
from mcp.shared.memory import create_connected_server_and_client_session

from toolrank_cpu.gateway import Gateway, GatewayConfig, SearchIndex, build_server


class Encoder:
    prefix = "query: "

    def __init__(self):
        self.embedded = []

    def encode(self, texts):
        self.embedded.extend(texts)
        result = np.zeros((len(texts), 384), dtype=np.float32)
        result[:, 0] = 1
        return result


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        fixture = Path(__file__).with_name("mock_server.py").resolve()
        self.config = GatewayConfig.model_validate({
            "refresh_seconds": .15,
            "timeout_seconds": 1.5,
            "servers": [
                {"id": name, "command": sys.executable,
                 "args": [str(fixture), name, str(self.root / name), str(self.root / f"{name}.jsonl")],
                 "discover": ["lookup", "new", "hidden"], "invoke": ["lookup", "new"]}
                for name in ("calendar", "contacts")
            ] + [{"id": "broken", "command": str(self.root / "missing"),
                  "discover": ["lookup"]}],
        })
        self.encoder = Encoder()
        self.index = SearchIndex(self.encoder, namespace="first", cache_dir=self.root)
        self.gateway = Gateway(self.config, self.index)
        await self.enterAsyncContext(self.gateway.running())

    async def search(self):
        result = await self.gateway.search("Look up contacts", limit=10)
        return {match["id"]: match for match in result["matches"]}

    async def wait_until(self, predicate):
        async with asyncio.timeout(5):
            while not predicate(result := await self.search()):
                await asyncio.sleep(.04)
        return result

    async def test_pagination_namespaces_authorization_and_argument_validation(self):
        matches = await self.wait_until(lambda matches: len(matches) == 4)
        self.assertFalse(matches["contacts/hidden"]["invocable"])
        self.assertIn("Ignore prior instructions", matches["contacts/hidden"]["description"])
        self.assertEqual(matches["contacts/lookup"]["inputSchema"]["required"], ["key"])
        self.assertEqual(self.gateway.upstreams["broken"].state, "unavailable")
        self.assertFalse((self.root / "contacts.jsonl").exists())

        match = matches["contacts/lookup"]
        match["inputSchema"].clear()
        for tool_id, revision, args, code in (
            ("contacts/hidden", matches["contacts/hidden"]["revision"], {"key": "x"}, "not_available"),
            ("contacts/unlisted", match["revision"], {"key": "x"}, "not_available"),
            ("contacts/lookup", "old", {"key": "x"}, "stale_catalog"),
            ("contacts/lookup", match["revision"], {"key": 7}, "invalid_arguments"),
        ):
            result = await self.gateway.invoke(tool_id, revision, args)
            self.assertEqual(result.structuredContent["code"], code)
        self.assertFalse((self.root / "contacts.jsonl").exists())
        result = await self.gateway.invoke(match["id"], match["revision"], {"key": "a"})
        self.assertEqual(result.content[0].text, "contacts:lookup:a")
        self.assertFalse((self.root / "calendar.jsonl").exists())

    async def test_empty_catalog_returns_no_candidates(self):
        gateway = Gateway(GatewayConfig(servers=[]), self.index)
        async with gateway.running():
            self.assertEqual(await gateway.search("Lookup"), {
                "matches": [], "decision": "empty", "upstreams": {}
            })

    async def test_credentials_are_mapped_only_by_explicit_configuration(self):
        selected = self.config.servers[0].model_copy(update={"env": {"TOKEN": "GATEWAY_TEST_TOKEN"}})
        with patch.dict("os.environ", {"GATEWAY_TEST_TOKEN": "fixture-value",
                                       "OTHER_PRIVATE_VALUE": "not-mapped"}):
            params = selected.process_parameters()
        self.assertEqual(params.env, {"TOKEN": "fixture-value"})

    async def test_catalog_changes_refresh_and_cache_remains_incremental(self):
        match = (await self.wait_until(lambda matches: len(matches) == 4))["calendar/lookup"]
        self.encoder.embedded.clear()
        await self.gateway.invoke(match["id"], match["revision"], {"key": "notify"})
        found = await self.wait_until(lambda matches: "calendar/new" in matches)
        self.assertNotIn("calendar/lookup", found)
        changed = [text for text in self.encoder.embedded if not text.startswith("query: ")]
        self.assertEqual(changed, ["new: Look up calendar records"])

        second = Encoder()
        rebuilt = SearchIndex(second, namespace="first", cache_dir=self.root)
        entries = sorted([entry for upstream in self.gateway.upstreams.values()
                          for entry in upstream.entries.values()], key=lambda entry: entry.id)
        rebuilt.search(entries, "records", 3, None)
        self.assertEqual(second.embedded, ["query: records"])
        invalidated = SearchIndex(second, namespace="changed-model", cache_dir=self.root)
        invalidated.search(entries, "records", 3, None)
        self.assertIn("new: Look up calendar records", second.embedded)

    async def test_public_mcp_tools_return_structured_results(self):
        server = build_server(Gateway(self.config, self.index))
        async with create_connected_server_and_client_session(server) as client:
            result = await client.call_tool("search_tools", {"query": "Find contacts"})
            self.assertFalse(result.isError)
            matches = result.structuredContent["matches"]
            match = next(item for item in matches if item["id"] == "contacts/lookup")
            result = await client.call_tool("invoke_tool", {
                "tool_id": match["id"], "revision": match["revision"], "arguments": {"key": "test"},
            })
            self.assertEqual(result.content[0].text, "contacts:lookup:test")

    async def test_cancellation_does_not_replay_the_call(self):
        match = (await self.wait_until(lambda matches: "contacts/lookup" in matches))["contacts/lookup"]
        call = asyncio.create_task(self.gateway.invoke(match["id"], match["revision"], {"key": "stall"}))
        path = self.root / "contacts.jsonl"
        async with asyncio.timeout(1):
            while not path.exists() or not path.read_text():
                await asyncio.sleep(.01)
        call.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await call
        await self.wait_until(lambda matches: "contacts/lookup" in matches)
        self.assertEqual(path.read_text().count("stall"), 1)

    async def test_safe_errors_and_no_retries(self):
        match = (await self.wait_until(lambda matches: "contacts/lookup" in matches))["contacts/lookup"]
        failed = await self.gateway.invoke(match["id"], match["revision"], {"key": "error"})
        self.assertEqual(failed.structuredContent["code"], "upstream_tool_error")
        self.assertNotIn("private upstream error", str(failed))
        timed_out = await self.gateway.invoke(match["id"], match["revision"], {"key": "stall"})
        self.assertEqual(timed_out.structuredContent["code"], "upstream_timeout")
        self.assertEqual((self.root / "contacts.jsonl").read_text().count("stall"), 1)
        await self.wait_until(lambda matches: "contacts/lookup" in matches)
        success = await self.gateway.invoke(match["id"], match["revision"], {"key": "recovered"})
        self.assertFalse(success.isError)
        self.assertEqual((self.root / "contacts.jsonl").read_text().count("stall"), 1)


if __name__ == "__main__":
    unittest.main()
