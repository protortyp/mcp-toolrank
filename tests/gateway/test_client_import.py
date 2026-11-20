import asyncio
import json
import os
from pathlib import Path
from stat import S_IMODE
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from test_http_upstreams import Endpoint, serve
from toolrank_cpu.gateway import GatewayConfig


def cli(*arguments):
    return [sys.executable, "-m", "toolrank_cpu.gateway", *map(str, arguments)]


class ClientImportTests(unittest.IsolatedAsyncioTestCase):
    async def test_import_then_inventory_real_stdio_and_bearer_http(self):
        directory = Path(self.enterContext(TemporaryDirectory()))
        source = directory / "client.json"
        destination = directory / "gateway.json"
        started = directory / "stdio-started"
        calls = directory / "calls"
        endpoint = Endpoint("remote", "fixture-secret")
        requests = []

        async def observed(scope, receive, send):
            if scope["type"] == "http":
                requests.append(scope["method"])
            await endpoint(scope, receive, send)

        url = await self.enterAsyncContext(serve(observed))
        child = Path(__file__).with_name("mock_server.py").resolve()
        program = f"from pathlib import Path; Path({str(started)!r}).touch(); import runpy, sys; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
        source.write_text(json.dumps({"mcpServers": {
            "local": {"command": sys.executable, "args": ["-c", program, str(child), "local",
                      str(directory / "state"), str(calls)], "cwd": str(directory),
                      "env": {"CHILD_TOKEN": "${env:TOOLRANK_IMPORT_TOKEN}"}},
            "remote": {"type": "http", "url": url, "headers": {"authorization": "Bearer ${TOOLRANK_IMPORT_TOKEN}"},
                       "tools": ["*"]},
            "disabled": {"command": "not-installed", "disabled": True},
        }}))
        original = source.read_bytes()
        env = os.environ | {"TOOLRANK_OFFLINE": "1", "TOOLRANK_IMPORT_TOKEN": "fixture-secret",
                            "XDG_CACHE_HOME": str(directory / "empty-cache")}
        imported = subprocess.run(cli(source, "--import-client-config", destination), cwd=directory,
                                  env=env, capture_output=True, timeout=10)
        self.assertEqual(imported.returncode, 0, imported.stderr)
        self.assertEqual(imported.stdout, b"")
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(requests, [])
        self.assertFalse(started.exists())
        self.assertNotIn(b"fixture-secret", destination.read_bytes())
        if os.name == "posix":
            self.assertEqual(S_IMODE(destination.stat().st_mode), 0o600)
        config = GatewayConfig.model_validate_json(destination.read_text())
        self.assertEqual(config.ranking, "bm25")
        self.assertEqual([entry.id for entry in config.servers], ["local", "remote"])
        self.assertEqual(config.servers[0].env, {"CHILD_TOKEN": "TOOLRANK_IMPORT_TOKEN"})
        self.assertEqual(config.servers[1].bearer_token_env, "TOOLRANK_IMPORT_TOKEN")
        self.assertTrue(all(not entry.discover and not entry.invoke for entry in config.servers))

        process = await asyncio.create_subprocess_exec(*cli(destination, "--list-upstream-tools"), cwd=directory,
                                                      env=env, stdout=asyncio.subprocess.PIPE,
                                                      stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(process.communicate(), 10)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        self.assertEqual(process.returncode, 0, err)
        report = json.loads(out)["upstreams"]
        self.assertEqual(report["local"]["tools"], ["hidden", "lookup", "unlisted"])
        self.assertEqual(report["remote"]["tools"], ["hidden", "lookup"])
        self.assertTrue(started.exists())
        self.assertFalse(calls.exists())
        self.assertEqual(endpoint.calls, [])
        self.assertEqual(endpoint.denied, 0)
        self.assertFalse((directory / "empty-cache").exists())

    async def test_strict_servers_format_and_no_overwrite(self):
        directory = Path(self.enterContext(TemporaryDirectory()))
        source = directory / "client.json"
        destination = directory / "gateway.json"
        source.write_text('{"servers":{"clock":{"type":"stdio","command":"python"},"off":{"enabled":false}}}')
        first = subprocess.run(cli(source, "--import-client-config", destination), check=True, capture_output=True,
                               timeout=10)
        self.assertEqual(first.stdout, b"")
        expected = destination.read_bytes()
        self.assertEqual([row["id"] for row in json.loads(expected)["servers"]], ["clock"])
        second = subprocess.run(cli(source, "--import-client-config", destination), capture_output=True, timeout=10)
        self.assertEqual(second.returncode, 2)
        self.assertIn(b"already exists", second.stderr)
        self.assertEqual(destination.read_bytes(), expected)

    async def test_rejects_unsafe_or_unsupported_settings_without_echoing_values(self):
        inputs = [
            '{"servers":{"one":{"command":"python","env":{"TOKEN":"private-marker"}}}}',
            '{"servers":{"one":{"type":"http","url":"https://example.com/mcp?token=private-marker"}}}',
            '{"servers":{"one":{"type":"http","url":"https://private-marker@example.com/mcp"}}}',
            '{"servers":{"one":{"type":"http","url":"https://example.com/mcp","headers":{"Authorization":"Bearer private-marker"}}}}',
            '{"servers":{"one":{"command":"python","cwd":"${workspaceFolder}"}}}',
            '{"servers":{"one":{"command":"python","sandboxEnabled":true}}}',
            '{"servers":{"one":{"command":"python","tools":["restricted"]}}}',
            '{"servers":{"one":{"type":"sse","url":"https://example.com/mcp"}}}',
            '{"servers":{"one":{"command":"python"},"one":{"command":"private-marker"}}}',
            '{"servers":{},"sandbox":{"filesystem":{}}}',
        ]
        directory = Path(self.enterContext(TemporaryDirectory()))
        source = directory / "client.json"
        destination = directory / "gateway.json"
        for number, data in enumerate(inputs):
            with self.subTest(number=number):
                source.write_text(data)
                proc = subprocess.run(cli(source, "--import-client-config", destination), capture_output=True,
                                      timeout=10)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(proc.stdout, b"")
                self.assertNotIn(b"private-marker", proc.stderr)
                self.assertFalse(destination.exists())
