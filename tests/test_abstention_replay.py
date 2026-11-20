import unittest

import numpy as np

from toolrank.abstention_replay import QueryCoverage, QueryScores, scope_metrics, wilson_95


class AbstentionReplayTests(unittest.TestCase):
    def test_rare_terms_weigh_more_and_unseen_terms_reduce_coverage(self):
        coverage = QueryCoverage(["get reading", "get forecast", "get station"])
        scores = coverage.scores("get get forecast", [0, 1])
        self.assertLess(scores[0], .5)
        self.assertAlmostEqual(scores[1], 1.0)
        self.assertLess(coverage.scores("forecast unknown", [1])[0], .5)
        self.assertEqual(coverage.scores("?", [1])[0], 0)

    def test_candidate_cosine_cutoff_differs_from_query_gate(self):
        names = np.array(["expected", "distractor"])
        supported = QueryScores({"expected"}, np.array([.5, .8]), np.array([.9, .8]), np.array([1., 1.]),
                                {"hybrid_max": .9})
        unsupported = QueryScores(set(), np.array([.1, .2]), np.array([.2, .3]), np.array([0., 0.]),
                                  {"hybrid_max": .3})
        data = [supported, unsupported]
        candidate = scope_metrics(data, names, "cosine", 5, .7)
        query = scope_metrics(data, names, "hybrid_max", 5, .7)
        lexical = scope_metrics(data, names, "bm25_overlap", 5)
        self.assertEqual(candidate["accepted_supported"], 1)
        self.assertEqual(candidate["supported_hits_at_5"], 0)
        self.assertEqual(query["supported_hits_at_5"], 1)
        self.assertEqual(lexical["accepted_unsupported"], 0)
        self.assertEqual(wilson_95(0, 16)[0], 0)
        self.assertGreater(wilson_95(0, 16)[1], .1)
