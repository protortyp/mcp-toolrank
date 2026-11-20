from math import log2
import unittest

from shortlist_pilot import api_payload, scores


class ShortlistPilotTests(unittest.TestCase):
    def test_graded_positions_and_unsupported_counts_are_independent(self):
        cases = [{"id": "supported", "expected": ["first", "second"]},
                 {"id": "unsupported", "expected": []}]
        rows, result = scores({"supported": ["other", "second"], "unsupported": ["other"]}, cases)
        self.assertAlmostEqual(rows[0], (1 / log2(3)) / (1 + 1 / log2(3)))
        self.assertEqual(result["hit_at_10"], 1)
        self.assertEqual(result["complete_at_10"], 0)
        self.assertEqual(result["supported"], 1)
        self.assertEqual(result["unsupported_accepted"], 1)

    def test_api_fixture_preserves_query_judgments_and_ids(self):
        payload = api_payload()
        ids = {f"{row['server']}/{row['tool']['name']}" for row in payload["tools"]}
        self.assertEqual(len(ids), 24)
        self.assertEqual(sum(bool(case["expected"]) for case in payload["cases"]), 72)
        self.assertEqual(sum(not case["expected"] for case in payload["cases"]), 24)
        self.assertTrue(all(set(case["expected"]) <= ids for case in payload["cases"]))
