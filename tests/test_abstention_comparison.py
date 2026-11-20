import unittest

import numpy as np

from toolrank.abstention_comparison import cutoff, metrics, signals


class AbstentionTests(unittest.TestCase):
    def test_tied_negatives_do_not_clear_cutoff(self):
        scores = np.array([.3, .3, .6], dtype=np.float32)
        supported = np.array([False, False, True])
        threshold = cutoff(scores, supported)
        result = metrics(scores, supported, supported, threshold)
        self.assertEqual(result["accepted_unsupported"], 0)
        self.assertEqual(result["supported_coverage"], 1.0)

    def test_margin_uses_cosine_and_standardized_score_is_finite(self):
        scores = np.array([[.1, .2, .8], [.4, .4, .4]])
        calculated = signals(scores, np.ones_like(scores))
        np.testing.assert_allclose(calculated["margin"], [.6, 0])
        np.testing.assert_allclose(calculated["hybrid_max"], [1, 1])
        self.assertTrue(np.isfinite(calculated["standardized_top"]).all())
