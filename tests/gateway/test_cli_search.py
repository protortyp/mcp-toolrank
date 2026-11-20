import asyncio
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from test_http_upstreams import Endpoint, serve


class SearchCliTests(unittest.IsolatedAsyncioTestCase):
    async def run_search(self, config, cache, *options):
        process = await asyncio.create_subprocess_exec(
            str(Path(sys.executable).with_name("toolrank-gateway")), str(config), *options,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=os.environ | {"TOOLRANK_OFFLINE": "1", "XDG_CACHE_HOME": str(cache),
                              "TOOLRANK_TEST_SEARCH_TOKEN": "fixture-private-auth"},
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        return process.returncode, stdout, stderr

    async def test_search_only_lists_allowed_matches_across_upstreams(self):
        directory = Path(self.enterContext(TemporaryDirectory()))
        endpoint = Endpoint("remote", "fixture-private-auth")
        url = await self.enterAsyncContext(serve(endpoint))
        cache = directory / "model-cache"
        calls = directory / "calls"
        config = directory / "gateway.json"
        servers = [
            {"id": "stdio", "command": sys.executable,
             "args": [str(Path(__file__).with_name("mock_server.py").resolve()), "local",
                      str(directory / "state"), str(calls)],
             "discover": ["lookup"], "invoke": ["lookup"]},
            {"id": "remote", "transport": "streamable_http", "url": url,
             "bearer_token_env": "TOOLRANK_TEST_SEARCH_TOKEN", "discover": ["lookup"]},
        ]
        config.write_text(json.dumps({"ranking": "bm25", "servers": servers}))

        code, stdout, stderr = await self.run_search(config, cache, "--search", "lookup", "--limit", "2")
        self.assertEqual((code, stderr), (0, b""))
        self.assertNotIn(b"fixture-private-auth", stdout)
        result = json.loads(stdout)
        self.assertEqual(result["upstreams"], {"stdio": "ready", "remote": "ready"})
        self.assertEqual(result["decision"], "ranked_unverified")
        self.assertEqual({row["id"]: row["invocable"] for row in result["matches"]},
                         {"remote/lookup": False, "stdio/lookup": True})

        code, stdout, stderr = await self.run_search(
            config, cache, "--search", "lookup", "--invocable-only", "--limit", "1"
        )
        self.assertEqual((code, stderr), (0, b""))
        self.assertEqual([row["id"] for row in json.loads(stdout)["matches"]], ["stdio/lookup"])

        code, stdout, stderr = await self.run_search(
            config, cache, "--search", "lookup", "--max-tool-bytes", "2"
        )
        self.assertEqual((code, stderr), (0, b""))
        result = json.loads(stdout)
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["decision"], "max_tool_bytes_exceeded")

        code, stdout, stderr = await self.run_search(config, cache, "--search", "xyzzy")
        self.assertEqual((code, stderr), (0, b""))
        result = json.loads(stdout)
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["decision"], "no_lexical_overlap")

        servers.append({"id": "offline", "command": str(directory / "missing"), "discover": ["lookup"]})
        config.write_text(json.dumps({"ranking": "bm25", "servers": servers}))
        code, stdout, stderr = await self.run_search(config, cache, "--search", "lookup")
        self.assertEqual((code, stderr), (1, b""))
        result = json.loads(stdout)
        self.assertEqual(result["upstreams"]["offline"], "unavailable")
        self.assertEqual(len(result["matches"]), 2)
        self.assertFalse(calls.exists())
        self.assertFalse(cache.exists())
        self.assertEqual(endpoint.calls, [])
        self.assertEqual(endpoint.denied, 0)

    async def test_invalid_options_fail_before_loading_config(self):
        directory = Path(self.enterContext(TemporaryDirectory()))
        for args in (("--search", ""), ("--search", "x", "--limit", "51"),
                     ("--search", "x", "--max-tool-bytes", "1"), ("--max-tool-bytes", "2")):
            with self.subTest(args=args):
                code, stdout, stderr = await self.run_search(directory / "missing.json", directory / "cache", *args)
                self.assertEqual((code, stdout), (2, b""))
                self.assertIn(b"--search", stderr)
                self.assertNotIn(b"Could not load", stderr)
