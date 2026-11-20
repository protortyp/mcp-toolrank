import unittest

import numpy as np

from examples.rank_onnx import CompactCatalog, evaluate, read_cases, read_catalog, top_indices


class FixedCatalog:
    def rank(self, request, *, limit):
        scores = {
            "both": [("calendar", 0.8), ("weather", 0.7)],
            "one": [("calendar", 0.4), ("weather", 0.3)],
            "none": [("weather", 0.6), ("calendar", 0.5)],
        }
        return [{"tool": {"name": name}, "score": score} for name, score in scores[request][:limit]]


class CatalogEvaluationTests(unittest.TestCase):
    def test_retrieval_metrics_and_threshold_are_measured_separately(self):
        cases = [
            ("both", {"calendar", "weather"}),
            ("one", {"weather"}),
            ("none", set()),
        ]
        result = evaluate(FixedCatalog(), cases, limit=1, min_score=0.5)
        self.assertEqual(result["metrics"], {"hit@1": 0.5, "recall@1": 0.25, "ndcg@1": 0.5})
        self.assertEqual(result["supported_coverage"], 0.5)
        self.assertEqual(result["unsupported_false_accept_rate"], 1.0)

    def test_compact_catalog_preserves_extreme_signed_dot_products(self):
        query = np.array([[1.0] * 384, [-1.0] * 384, [0.0] * 384], dtype=np.float32)
        docs = np.repeat(query, 33, axis=0)
        catalog = CompactCatalog((docs[:64], docs[64:]), len(docs), threads=1)
        np.testing.assert_allclose(catalog.scores(query), query @ docs.T, atol=1e-4)
        self.assertEqual(catalog.vectors.nbytes + catalog.scales.nbytes, len(docs) * (384 + 4))

    def test_partial_top_k_preserves_score_order_and_catalog_order_for_ties(self):
        scores = np.array([0.6, 0.8, 0.8, 0.6, 0.9, 0.2, 0.8])
        for limit in range(1, len(scores) + 2):
            np.testing.assert_array_equal(top_indices(scores, limit),
                                          np.argsort(-scores, kind="stable")[:limit])

    def test_json_rpc_catalog_and_case_validation(self):
        tools = read_catalog({"result": {"tools": [{"name": "calendar"}]}})
        self.assertEqual(tools, [{"name": "calendar"}])
        self.assertEqual(read_cases([{"query": "none", "relevant": []}], {"calendar"}),
                         [("none", set())])
        with self.assertRaisesRegex(ValueError, "outside the catalog"):
            read_cases([{"query": "forecast", "relevant": ["weather"]}], {"calendar"})


if __name__ == "__main__":
    unittest.main()
