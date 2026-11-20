import unittest

from toolrank_cpu.gateway import limit_tool_bytes

from run import catalog, interleave, schema_bytes


class MixedCatalogTests(unittest.TestCase):
    def test_catalog_preserves_provider_names_and_station_path_schema(self):
        entries = catalog()
        self.assertEqual(len(entries), 93)
        self.assertEqual(len({entry.server for entry in entries}), 7)
        station = next(entry for entry in entries if entry.id == "nws/weather__obs_station")
        self.assertIn("stationId", station.tool.inputSchema["properties"])
        self.assertIn("stationId", station.tool.inputSchema["required"])

    def test_clause_merge_keeps_ranked_alternation_and_omits_duplicates(self):
        self.assertEqual(interleave([1, 3, 4], [1, 2, 5], 4), [1, 3, 2, 4])

    def test_budget_counts_utf8_and_array_delimiters(self):
        pieces = [{"name": "東京"}, {"name": "résumé"}]
        self.assertEqual(limit_tool_bytes(pieces, schema_bytes(pieces)), pieces)
        self.assertEqual(limit_tool_bytes(pieces, schema_bytes(pieces) - 1), pieces[:1])
        self.assertEqual(limit_tool_bytes(pieces, schema_bytes(pieces[:1]) - 1), [])
