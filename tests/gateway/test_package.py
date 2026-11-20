from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from toolrank_cpu import cli
from toolrank_cpu.catalog_lexical import CatalogLexicalIndex


class PackageTests(unittest.TestCase):
    def test_model_free_gateway_does_not_import_model_libraries(self):
        program = """
import sys
from toolrank_cpu.gateway import Gateway, GatewayConfig
Gateway.from_config(GatewayConfig(servers=[], ranking='bm25'))
assert not {'onnxruntime', 'tokenizers'}.intersection(sys.modules)
"""
        with patch.dict("os.environ", {"TOOLRANK_OFFLINE": "1"}):
            subprocess.run([sys.executable, "-c", program], check=True, timeout=10)

    def test_bm25_only_matches_hybrid_bm25_component(self):
        tools = ["report: Download an account statement", "statement: Update a report", "clock: read time"]
        index = CatalogLexicalIndex(tools)
        bm25 = CatalogLexicalIndex(tools, bm25_only=True)
        for query in ("account report", "time", "nohits"):
            self.assertEqual(bm25.bm25_scores(query).tolist(), index.scores(query)[1].tolist())

    def test_checksums_reject_corrupt_bytes(self):
        with TemporaryDirectory() as temp:
            fingerprint = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
            cached = Path(temp) / "data"
            cached.write_bytes(b"hello")
            self.assertTrue(cli.matches(cached, fingerprint))
            cached.write_bytes(b"changed")
            self.assertFalse(cli.matches(cached, fingerprint))

    def test_offline_does_not_attempt_download(self):
        with TemporaryDirectory() as temp, patch.dict("os.environ", {
            "XDG_CACHE_HOME": temp, "TOOLRANK_OFFLINE": "1",
        }), patch.object(cli, "urlopen") as download:
            with self.assertRaisesRegex(ValueError, "verified cached file"):
                cli.checkpoint()
            download.assert_not_called()
