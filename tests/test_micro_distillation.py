import unittest

import torch

from toolrank.train_micro_distillation import keys_and_judgments, softened_kl


class DistillationTests(unittest.TestCase):
    def test_shared_names_and_provider_are_masked_without_losing_positives(self):
        items = {"a": {"name": "find", "repository": "provider-a"},
                 "b": {"name": "find", "repository": "provider-b"},
                 "c": {"name": "export", "repository": "provider-a"},
                 "d": {"name": "export", "repository": "provider-c"},
                 "e": {"name": "check", "repository": "provider-c"}}
        for positives, expected in ((["a"], [True, False, False, True, True]),
                                    (["a", "c"], [True, False, True, False, True])):
            keys, positive, allowed = keys_and_judgments({"tools": items, "queries": [{"positives": positives}]})
            self.assertEqual([key for key, keep in zip(keys, positive[0]) if keep], positives)
            self.assertEqual(allowed[0].tolist(), expected)

    def test_distillation_excludes_ambiguous_negative_scores_and_gradients(self):
        student = torch.tensor([[.5, .2, -10.]], requires_grad=True)
        eligible = torch.tensor([[True, True, False]])
        first = softened_kl(student, torch.tensor([[.1, .4, 100.]]), eligible, .1)
        second = softened_kl(student, torch.tensor([[.1, .4, -100.]]), eligible, .1)
        self.assertEqual(first.item(), second.item())
        first.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertEqual(student.grad[0, 2].item(), 0)
