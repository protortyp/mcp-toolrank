import json
from types import SimpleNamespace
import unittest

import numpy as np

from verify_capability import decisions


class VerifierTests(unittest.TestCase):
    def test_verified_counts_distinguish_rejection_from_retrieval(self):
        rows = [{"id": "accepted", "expected": "goal"},
                {"id": "rejected", "expected": "goal"},
                {"id": "unsupported", "expected": None}]
        ranker = SimpleNamespace(tools=[{"name": "other"}, {"name": "goal"}])
        choices = [np.array([0, 1])] * 3
        values = np.array([[0.1, .9], [.1, .2], [.8, .3]], dtype=np.float32)
        report = json.loads(json.dumps(decisions(rows, ranker, choices, values, .5)))
        self.assertEqual(report["supported_hit_at_5"], 1)
        self.assertEqual(report["supported_hit_at_1"], 1)
        self.assertEqual(report["candidate_recall_count"], 2)
        self.assertEqual(report["accepted_unsupported"], 1)
        self.assertEqual(report["missed_supported_ids"], ["rejected"])
