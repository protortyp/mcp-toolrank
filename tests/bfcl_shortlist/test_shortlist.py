import unittest

import numpy as np

from run import aggregate, json_length, measure, messages, select


class ShortlistTests(unittest.TestCase):
    def test_user_history_preserves_order_and_ignores_other_roles(self):
        question = [[{"role": "user", "content": "First"},
                     {"role": "assistant", "content": "Response"},
                     {"role": "user", "content": "Second"}]]
        self.assertEqual(messages(question), "First\nSecond")

    def test_empty_unsupported_menu_has_no_exposure(self):
        row = measure([], set(), [], json_length([]))
        self.assertEqual(aggregate([row])["accepted_unsupported"], 0)

    def test_half_byte_policy_retains_full_menu_below_savings_target(self):
        tools = [{"name": f"tool_{i}"} for i in range(6)]
        order = np.arange(5, -1, -1)
        np.testing.assert_array_equal(select(tools, order, min_savings=.5), order)
        self.assertEqual(len(select(tools, order)), 5)

    def test_complete_retrieval_requires_all_gold_functions(self):
        tools = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        row = measure(tools, {"a", "b"}, [0, 2], json_length(tools))
        self.assertEqual(aggregate([row])["complete"], 0)
        self.assertEqual(aggregate([row])["hit"], 1)
