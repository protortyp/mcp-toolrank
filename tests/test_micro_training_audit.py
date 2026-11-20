import unittest

import numpy as np

from toolrank.audit_micro_errors import audit_mask, judgment_outcomes, paired_misses


class MicroTrainingAuditTests(unittest.TestCase):
    def test_same_repository_hits_are_distinct_from_other_unjudged_hits(self):
        matrix = np.array([[1, 2, 0], [2, 3, 4], [5, 1, 0]])
        categories, hit5 = judgment_outcomes(matrix, [{0}, {0, 1}, {0}], ["source", "source", "other"])
        self.assertEqual(categories, ["unjudged_same_repository", "unjudged_other_repository", "judged_positive"])
        self.assertEqual(hit5, [True, True, True])
        self.assertEqual(paired_misses([False, False, True, True], [False, True, False, True]),
                         {"both_miss": 1, "comparison_only_hit": 1, "candidate_only_hit": 1, "both_hit": 1})

    def test_unjudged_review_pairs_stay_distinct_from_supervised_negatives(self):
        training = {
            "tools": {
                "a": {"repository": "shared", "name": "a"},
                "b": {"repository": "shared", "name": "b"},
                "c": {"repository": "other", "name": "c"},
            },
            "queries": [{"id": "row-1", "positives": ["a"]}, {"id": "row-2", "positives": ["a"]}],
        }
        report, requests = audit_mask(training, np.array([[.9, 1, .5], [.8, .6, .9]]))
        self.assertEqual(report["top_unjudged_masked"], 1)
        self.assertEqual(report["queries_with_unjudged_same_repository"], 2)
        self.assertEqual(report["top_unjudged_same_repository_above_best_positive"], 1)
        self.assertEqual(report["top_unjudged_above_best_positive"], 2)
        self.assertEqual(requests[0]["judged_sibling"], "a")
        self.assertEqual(requests[0]["unjudged_candidate"], "b")
        self.assertEqual(len(requests), 1)
