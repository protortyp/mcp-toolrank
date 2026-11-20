import unittest

import numpy as np

from evaluate import confidence, evaluate, pilot_gate


class PilotTests(unittest.TestCase):
    def test_margin_uses_highest_two_cosines(self):
        features = confidence(np.array([.2, .8, .3]), np.array([.3, .8, .9]))
        self.assertAlmostEqual(features["margin"], .5)
        self.assertEqual(features["hybrid_max"], .9)

    def test_cosine_cutoff_removes_low_cosine_tool_even_when_it_leads_hybrid(self):
        tools = [{"name": "one"}, {"name": "two"}]
        samples = [
            ({"id": "p", "expected": "one", "features": {"cosine": .8}},
             {"dense": np.array([.1, .8]), "hybrid": np.array([.9, .8])}),
            ({"id": "n", "expected": None, "features": {"cosine": .2}},
             {"dense": np.array([.2, .1]), "hybrid": np.array([.2, .1])}),
        ]
        outcome = evaluate(samples, tools, "hybrid", "cosine", .7)
        self.assertEqual(outcome["accepted_unsupported"], 0)
        self.assertEqual(outcome["accepted_supported"], 1)
        self.assertEqual(outcome["supported_hit_at_5"], 0)

    def test_gate_needs_both_rejection_and_coverage(self):
        hybrid = {"supported_hit_at_5": 19, "supported": 20, "tool_json_reduction": .85}
        cutoff = {"accepted_unsupported": 0, "unsupported": 20, "accepted_supported": 16, "supported": 20}
        gate = pilot_gate({"hybrid": hybrid, "hybrid_cosine_cutoff": cutoff})
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["supported_coverage_at_least_85_percent"])
