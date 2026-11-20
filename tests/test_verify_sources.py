import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from toolrank import verify_sources
from toolrank.core import read_json, write_json


class SourceVerificationTests(unittest.TestCase):
    def test_excludes_targets_without_textual_descriptions(self):
        server = {
            "server_name": "sample",
            "server_info": {"repository_url": "https://github.com/example/sample"},
            "remote_server_response": {"tool_names": ["lookup"]},
        }
        rows = []
        for index, spec in enumerate(({}, {"description": None}, {"description": 5})):
            rows.append({
                "uuid": str(index),
                "question": "Find a sample record.",
                "target_tools": "sample-lookup",
                "available_tools": json.dumps([{"function": {"name": "sample-lookup", **spec}}]),
                "metadata": json.dumps({"mcp_servers": [server]}),
            })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "fixture.parquet"
            pq.write_table(pa.Table.from_pylist(rows), archive)
            write_json(root / "fixtures.json", {"mcpServers": {}})
            write_json(root / "reserved-families.json", {"names": [], "repository_slugs": []})
            write_json(root / "manifest.json", {"example/sample": {"spdx": "MIT", "observed_default_head": "test"}})
            with patch.object(verify_sources, "ETOM_ROOT", root), patch.object(
                verify_sources, "ATLAS_ROOT", root / "fixtures.json"
            ):
                verify_sources.run([archive], root / "report.json", root / "rows.json", root / "manifest.json")
            report = read_json(root / "report.json")
            self.assertEqual(report["counts"]["target_mentions_without_text_descriptions"], 3)
            self.assertEqual(read_json(root / "rows.json"), [])


if __name__ == "__main__":
    unittest.main()
