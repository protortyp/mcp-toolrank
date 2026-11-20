import unittest

from agent import classify, summary
from native_agent import NativeModel, validate_batch


class AgentEvaluatorTests(unittest.TestCase):
    def test_retrieval_and_selection_failures_are_distinct(self):
        expected = [{"id": "catalog/gold", "args": {"id": "1"}}]
        proposed = {"tool_id": "catalog/other", "arguments": {}}
        self.assertEqual(classify(proposed, expected, {"catalog/other"})[0], "correct_tool_omitted")
        self.assertEqual(classify(proposed, expected, {"catalog/gold", "catalog/other"})[0],
                         "correct_tool_not_selected")

    def test_argument_validation_rejects_extras_and_wrong_types(self):
        expected = [{"id": "catalog/gold", "args": {"id": 1}}]
        visible = {"catalog/gold"}
        self.assertEqual(classify({"tool_id": "catalog/gold", "arguments": {"id": "1"}},
                                  expected, visible)[0], "incorrect_arguments")
        self.assertEqual(classify({"tool_id": "catalog/gold", "arguments": {"id": 1}},
                                  expected, visible), (None, expected[0]))

    def test_unsupported_request_counts_only_abstention_as_success(self):
        self.assertEqual(classify({"tool_id": None, "arguments": {}}, [], {"catalog/read"}), (None, None))
        self.assertEqual(classify({"tool_id": "catalog/read", "arguments": {}}, [], {"catalog/read"})[0],
                         "wrong_tool_call")
        row = {"success": True, "supported": False, "failure": None,
               "prompt_tokens": 3, "tool_schema_tokens": 2, "elapsed_ms": 1}
        self.assertEqual(summary([row])["unsupported_rejected"], 1)

    def test_native_action_handles_wire_formats_and_preserves_identity(self):
        names = {"catalog__read": "catalog/read"}
        for arguments in ('{"id":1}', {"id": 1}):
            function = {"name": "catalog__read", "arguments": arguments}
            call = {"type": "function", "function": function}
            choice = {"message": {"tool_calls": [call]}, "finish_reason": "tool_calls"}
            self.assertEqual(NativeModel.action(choice, names), {"tool_id": "catalog/read", "arguments": {"id": 1}})
        choice["message"]["tool_calls"].append(call)
        self.assertIsNone(NativeModel.action(choice, names))
        self.assertEqual(len(NativeModel.actions(choice, names)), 2)

    def test_native_action_distinguishes_abstention_from_truncation(self):
        self.assertEqual(NativeModel.action({"message": {}, "finish_reason": "stop"}, {}),
                         {"tool_id": None, "arguments": {}})
        self.assertIsNone(NativeModel.action({"message": {}, "finish_reason": "length"}, {}))

    def test_batch_validates_every_call_before_dispatch(self):
        expected = [{"id": "catalog/first", "args": {"id": "1"}}, {"id": "provider/second", "args": {}}]
        first = {"tool_id": "catalog/first", "arguments": {"id": "1"}}
        second = {"tool_id": "provider/second", "arguments": {}}
        visible = {row["id"] for row in expected}
        self.assertEqual(validate_batch([first, second], expected, visible), (None, expected))
        self.assertEqual(validate_batch([first, second], expected[:1], visible), ("wrong_tool_call", []))
        self.assertEqual(validate_batch([first, second], expected, {"catalog/first"}), ("tool_outside_shortlist", []))
