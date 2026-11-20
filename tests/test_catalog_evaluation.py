import unittest

from examples.rank_onnx import evaluate, read_cases, read_catalog


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

    def test_json_rpc_catalog_and_case_validation(self):
        tools = read_catalog({"result": {"tools": [{"name": "calendar"}]}})
        self.assertEqual(tools, [{"name": "calendar"}])
        self.assertEqual(read_cases([{"query": "none", "relevant": []}], {"calendar"}),
                         [("none", set())])
        with self.assertRaisesRegex(ValueError, "outside the catalog"):
            read_cases([{"query": "forecast", "relevant": ["weather"]}], {"calendar"})


if __name__ == "__main__":
    unittest.main()
