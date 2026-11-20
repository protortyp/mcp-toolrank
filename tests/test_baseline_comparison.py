import unittest

import numpy as np

from toolrank.baseline_comparison import measure, paired_interval, rankings, top_indices


class BaselineComparisonTests(unittest.TestCase):
    def test_bge_and_toolrank_share_both_lexical_terms(self):
        dense = np.array([[.1, .2, .3]])
        tfidf = np.array([[.3, .2, .1]])
        bm25 = np.array([[1., 0., .5]])
        scores = dict(rankings({"bge": dense, "toolrank": dense.copy()}, tfidf, bm25))
        np.testing.assert_allclose(scores["bge-hybrid"], dense + .4 * tfidf + .15 * bm25)
        np.testing.assert_equal(scores["bge-hybrid"], scores["toolrank-hybrid"])
        np.testing.assert_equal(scores["tfidf"], tfidf)

    def test_hit_recall_and_completeness_measure_different_outcomes(self):
        ranking = ["a", "q", "c", "d", "e", "b"]
        result = measure(ranking, {"a": 2, "b": 1})
        self.assertEqual(result["hit@1"], 1)
        self.assertEqual(result["recall@5"], .5)
        self.assertEqual(result["complete@5"], 0)
        self.assertEqual(result["complete@10"], 1)

    def test_catalog_top_ten_matches_full_stable_sort_with_ties(self):
        rng = np.random.default_rng(37)
        dense = rng.standard_normal((8, 44_453), dtype=np.float32)
        for matrix in (dense, np.round(dense), np.zeros((1, 44_453))):
            expected = np.argsort(-matrix, axis=1, kind="stable")[:, :10]
            actual = np.stack([top_indices(row, 10) for row in matrix])
            np.testing.assert_equal(actual, expected)

    def test_paired_interval_preserves_constant_differences(self):
        interval = paired_interval([1, 1, 0], [1, 1, 0], ["a", "a", "b"])
        self.assertEqual(interval, {"delta": 0, "ci95": [0, 0], "clusters": 2})
