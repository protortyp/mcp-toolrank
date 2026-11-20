import unittest

import torch

from toolrank.family_weighting import family_weights, gate_result
from toolrank.train_modern import multi_positive_loss


class FamilyWeightingTests(unittest.TestCase):
    def test_frequent_families_receive_less_weight_per_row(self):
        rows = [{"families": ["common"]}] * 16 + [{"families": ["rare"]}]
        weights = family_weights(rows, exponent=.5, cap=3)
        self.assertAlmostEqual(weights.mean().item(), 1, places=6)
        self.assertTrue(torch.equal(weights[:16], weights[0].expand(16)))
        self.assertGreater(weights[-1].item(), weights[0].item())
        self.assertLess(weights[-1].item() / weights[0].item(), 4)

    def test_weighted_loss_preserves_the_uniform_case(self):
        logits = torch.tensor([[2., 0.], [0., 1.]], requires_grad=True)
        positives = torch.tensor([[True, False], [True, False]])
        allowed = torch.ones_like(positives)
        unweighted = multi_positive_loss(logits, positives, allowed)
        self.assertTrue(torch.equal(unweighted, multi_positive_loss(logits, positives, allowed, torch.ones(2))))
        weighted = multi_positive_loss(logits, positives, allowed, torch.tensor([.5, 1.5]))
        expected = (torch.nn.functional.softplus(torch.tensor(-2.)) * .5 +
                    torch.nn.functional.softplus(torch.tensor(1.)) * 1.5) / 2
        torch.testing.assert_close(weighted, expected)
        weighted.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_gate_rejects_a_decline_in_any_expanded_seed(self):
        def catalog(gains):
            return {"weighted_minus_control": {
                str(i): {"hit@5": {"delta": gain}, "ndcg@10": {"delta": .006}}
                for i, gain in enumerate(gains)
            }}

        values = {"compact": catalog([.02, .02, .02]), "expanded": catalog([.03, .03, -.001])}
        gate = {"min_mean_expanded_macro_hit5_gain": .01, "min_mean_expanded_macro_ndcg10_gain": .005,
                "max_mean_compact_macro_ndcg10_loss": .005, "min_expanded_macro_hit5_gain_per_seed": 0}
        result = gate_result(values, gate)
        self.assertFalse(result["passes"])
        self.assertEqual(result["checks"], {"expanded_hit5": True, "expanded_ndcg": True,
                                            "compact_ndcg": True, "seed_hit5": False})
