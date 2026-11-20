import unittest

import numpy as np

from toolrank.selector import ToolSelector


class FixedEncoder:
    def encode_document(self, texts, *, normalize_embeddings):
        return np.array([
            (0.8, 0.6) if text.startswith("mail:") else (0.7, np.sqrt(0.51))
            for text in texts
        ])

    def encode_query(self, query, *, normalize_embeddings):
        return np.array([1.0, 0.0])


class HybridSelectorTests(unittest.TestCase):
    def test_lexical_score_changes_order_and_min_score_still_filters_cosine(self):
        tools = [
            {"name": "mail", "description": "Retrieve new mail"},
            {"name": "calendar", "description": "Get upcoming calendar events"},
        ]
        query = "upcoming calendar events"
        dense = ToolSelector(FixedEncoder(), tools)
        hybrid = ToolSelector(FixedEncoder(), tools, lexical_weight=0.4)
        self.assertEqual(dense.rank(query, k=1)[0].tool["name"], "mail")
        match = hybrid.rank(query, k=1)[0]
        self.assertEqual(match.tool["name"], "calendar")
        self.assertAlmostEqual(match.cosine, 0.7)
        self.assertEqual(hybrid.rank(query, k=1, min_score=0.75)[0].tool["name"], "mail")

    def test_bm25_tool_name_changes_order_without_altering_cosine_cutoff(self):
        tools = [{"name": "mail", "description": "Retrieve new mail"},
                 {"name": "calendar", "description": "Get upcoming events"}]
        selector = ToolSelector(FixedEncoder(), tools, bm25_weight=0.15)
        first = selector.rank("calendar", k=1)[0]
        self.assertEqual(first.tool["name"], "calendar")
        self.assertAlmostEqual(first.cosine, 0.7)
        self.assertEqual(selector.rank("calendar", k=1, min_score=0.75)[0].tool["name"], "mail")

    def test_bm25_empty_lexical_catalog_falls_back_to_dense(self):
        selector = ToolSelector(FixedEncoder(), [{"name": "_"}], bm25_weight=0.15)
        self.assertEqual(selector.rank("calendar", k=1)[0].tool["name"], "_")

    def test_catalog_without_lexical_terms_can_still_use_dense_vectors(self):
        selector = ToolSelector(FixedEncoder(), [{"name": "_"}], lexical_weight=0.4)
        self.assertEqual(selector.rank("a request")[0].tool["name"], "_")


if __name__ == "__main__":
    unittest.main()
