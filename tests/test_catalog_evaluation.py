import unittest

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from examples.catalog_lexical import CatalogLexicalIndex
from examples.rank_onnx import (
    CompactCatalog, OnnxToolRank, allowed_negative_exceedances, calibrate, evaluate, read_cases,
    read_catalog, top_indices,
)

from toolrank.lexical import Bm25Index, lexical_text


class FixedCatalog:
    def rank(self, request, *, limit, min_score=None):
        scores = {
            "both": [("calendar", 0.8), ("weather", 0.7)],
            "one": [("calendar", 0.4), ("weather", 0.3)],
            "none": [("weather", 0.6), ("calendar", 0.5)],
        }
        matches = [{"tool": {"name": name}, "score": score}
                   for name, score in scores[request] if min_score is None or score >= min_score]
        return matches[:limit]


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

    def test_onnx_hybrid_filters_by_cosine_before_applying_the_limit(self):
        ranker = object.__new__(OnnxToolRank)
        ranker.tools = [{"name": "calendar"}, {"name": "mail"}]
        ranker.docs = np.array([[.7, 0], [.8, 0]], dtype=np.float32)
        ranker.catalog = None
        ranker.prefix = ""
        ranker.encode = lambda texts: np.array([[1, 0]], dtype=np.float32)
        ranker.lexical_index = CatalogLexicalIndex(["calendar: List events", "mail: Check messages"])
        self.assertEqual(ranker.rank("calendar", limit=1)[0]["tool"]["name"], "calendar")
        allowed = ranker.rank("calendar", limit=1, min_score=.75)
        self.assertEqual(allowed[0]["tool"]["name"], "mail")
        self.assertAlmostEqual(allowed[0]["score"], .8)
        self.assertEqual(ranker.rank("calendar", limit=1, min_score=1.1), [])
        result = evaluate(ranker, [("calendar", {"mail"})], limit=1, min_score=.75)
        self.assertEqual(result["accepted_supported"], 1)
        self.assertEqual(result["metrics"]["recall@1"], 0)

    def test_partial_top_k_preserves_score_order_and_catalog_order_for_ties(self):
        scores = np.array([0.6, 0.8, 0.8, 0.6, 0.9, 0.2, 0.8])
        for limit in range(1, len(scores) + 2):
            np.testing.assert_array_equal(top_indices(scores, limit),
                                          np.argsort(-scores, kind="stable")[:limit])

    def test_onnx_lexical_index_matches_python_selector_scores(self):
        documents = ["getLocalTime: Time in São Paulo", "find_calendar: Read the calendar remotely",
                     '{"name": "time_zone_list", "description": "Read the time zones"}', "_"]
        queries = ["Time in São Paulo", "read my calendar", "x a ?, ZZ unknown"]
        vectorizer = TfidfVectorizer(preprocessor=lexical_text, ngram_range=(1, 2))
        reference_docs = vectorizer.fit_transform(documents)
        expected_tfidf = (vectorizer.transform(queries) @ reference_docs.T).toarray()
        expected_bm25 = Bm25Index(documents).scores(queries)
        index = CatalogLexicalIndex(documents)
        for row, query in enumerate(queries):
            tfidf, bm25 = index.scores(query)
            np.testing.assert_allclose(tfidf, expected_tfidf[row], atol=1e-12)
            np.testing.assert_allclose(bm25, expected_bm25[row], atol=3e-7)
        np.testing.assert_array_equal(CatalogLexicalIndex(["_"]).blend("empty", np.array([.3])), [.3])

    def test_order_statistic_respects_the_requested_false_accept_bound(self):
        self.assertIsNone(allowed_negative_exceedances(28, .1, .95))
        self.assertEqual(allowed_negative_exceedances(29, .1, .95), 0)
        self.assertEqual(allowed_negative_exceedances(100, .1, .95), 4)

    def test_calibration_excludes_float32_ties(self):
        cases = [("0.75", set())] * 29 + [("0.8", {"calendar"})]
        result = calibrate(cases, score_requests=lambda requests: list(map(float, requests)),
                           max_false_accept=.1, confidence=.95)
        self.assertEqual(result["min_score"], float(np.nextafter(np.float32(.75), np.float32(np.inf))))
        self.assertEqual(result["accepted_no_match_on_calibration"], 0)
        self.assertEqual(result["accepted_answerable_on_calibration"], 1)

    def test_too_few_calibration_negatives_does_not_score_requests(self):
        def unused_scores(requests):
            self.fail("calibration must check sample size before encoding")

        result = calibrate([("unknown", set())], score_requests=unused_scores,
                           max_false_accept=.1, confidence=.95)
        self.assertEqual(result["status"], "insufficient_no_match_cases")
        self.assertIsNone(result["min_score"])
        self.assertEqual(result["minimum_no_match_cases"], 29)

    def test_json_rpc_catalog_and_case_validation(self):
        tools = read_catalog({"result": {"tools": [{"name": "calendar"}]}})
        self.assertEqual(tools, [{"name": "calendar"}])
        self.assertEqual(read_cases([{"query": "none", "relevant": []}], {"calendar"}),
                         [("none", set())])
        with self.assertRaisesRegex(ValueError, "outside the catalog"):
            read_cases([{"query": "forecast", "relevant": ["weather"]}], {"calendar"})


if __name__ == "__main__":
    unittest.main()
