from contextlib import asynccontextmanager
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mcp.types import Tool
import numpy as np

from toolrank_cpu.gateway import CalibratedCutoff, Entry, Gateway, GatewayConfig, SearchIndex
from toolrank_cpu.gateway_calibration import run_labeled


class Encoder:
    prefix = "query: "

    def encode(self, texts):
        named = {"clock: Read time": [1., 0.], "status: Git state": [0., 1.],
                 "query: time": [1., 0.], "query: status": [0., 1.]}
        return np.array([named.get(text, [.6, .8]) for text in texts], dtype=np.float32)


class CalibrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = GatewayConfig.model_validate({"servers": [{"id": "local", "command": "fixture",
                                                                 "discover": ["clock", "status"], "invoke": ["clock"]}]})
        self.gateway = Gateway(self.config, SearchIndex(Encoder(), namespace="fixture"))
        self.upstream = self.gateway.upstreams["local"]
        self.upstream.state = "ready"
        for name, desc, allowed in (("clock", "Read time", True), ("status", "Git state", False)):
            entry = Entry("local", Tool(name=name, description=desc, inputSchema={"type": "object"}), allowed, "v1")
            self.upstream.entries[entry.id] = entry

    @asynccontextmanager
    async def running_fixture(self):
        yield self.gateway

    async def test_fit_and_evaluate_on_exact_search_scope(self):
        labels = [{"query": f"unknown {i}", "relevant": []} for i in range(29)]
        labels.append({"query": "time", "relevant": ["local/clock"]})
        with TemporaryDirectory() as temp:
            data = Path(temp) / "labels.json"
            data.write_text(json.dumps(labels))
            opts = SimpleNamespace(calibrate=data, evaluate=None, invocable_only=True, max_false_accept=.10,
                                   confidence=.95, limit=5)
            with patch.object(self.gateway, "running", self.running_fixture):
                report = await run_labeled(self.gateway, opts)
                self.assertEqual(report["eligible_tools"], 1)
                self.assertEqual(report["accepted_no_match_on_calibration"], 0)
                self.assertGreater(report["min_score"], .6)
                self.assertLess(report["min_score"], .8)
                self.gateway.calibration = CalibratedCutoff.model_validate(report)
                data.write_text(json.dumps([{"query": "time", "relevant": ["local/clock"]},
                                            {"query": "status", "relevant": []}]))
                opts.calibrate, opts.evaluate = None, data
                results = await run_labeled(self.gateway, opts)
                self.assertEqual(results["supported_coverage"], 1.0)
                self.assertEqual(results["unsupported_false_accept_rate"], 0.0)

    async def test_scope_mismatch_and_catalog_drift_fail_closed(self):
        scope = self.gateway.calibration_scope(self.gateway.catalog_entries(), True)
        self.gateway.calibration = CalibratedCutoff(status="calibrated", schema_version=1, scope=scope, min_score=.7)
        self.assertEqual((await self.gateway.search("time"))["decision"], "calibration_scope_mismatch")
        self.assertEqual((await self.gateway.search("time", invocable_only=True))["decision"], "above_threshold")
        self.upstream.entries["local/status"] = replace(self.upstream.entries["local/status"], revision="v2")
        self.assertEqual((await self.gateway.search("time", invocable_only=True))["decision"], "above_threshold")
        self.upstream.entries["local/clock"] = replace(self.upstream.entries["local/clock"], revision="v2")
        result = await self.gateway.search("time", invocable_only=True)
        self.assertEqual(result["decision"], "calibration_scope_mismatch")
        self.assertEqual(result["matches"], [])

    async def test_scope_checked_after_scoring(self):
        scope = self.gateway.calibration_scope(self.gateway.catalog_entries(), True)
        self.gateway.calibration = CalibratedCutoff(status="calibrated", schema_version=1, scope=scope, min_score=.7)
        search = self.gateway.index.search

        def score_then_change(*args, **kwargs):
            matches = search(*args, **kwargs)
            self.upstream.entries["local/status"] = replace(self.upstream.entries["local/status"], invocable=True)
            return matches

        with patch.object(self.gateway.index, "search", score_then_change):
            response = await self.gateway.search("time", invocable_only=True)
        self.assertEqual(response["decision"], "calibration_scope_mismatch")
        self.assertEqual(response["matches"], [])
