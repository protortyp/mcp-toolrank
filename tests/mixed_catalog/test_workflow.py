import unittest
from unittest.mock import AsyncMock, Mock

from jsonschema import Draft202012Validator

from native_agent import NativeModel
from workflow import compact_schema, evaluate, normalize_defaults


class WorkflowTests(unittest.TestCase):
    def test_compaction_preserves_constraints_property_names_and_literal_values(self):
        schema = {
            "description": "Object documentation.",
            "properties": {
                "description": {"description": "String documentation.", "type": "string", "minLength": 3},
                "payload": {"const": {"description": "Literal value"}},
                "units": {"allOf": [{"enum": ["us", "si"], "examples": ["us"]}], "default": "us"},
            },
            "additionalProperties": False,
            "required": ["description", "payload"],
        }
        compact = compact_schema(schema)
        self.assertNotIn("description", compact)
        self.assertEqual(compact["properties"]["description"], {"type": "string", "minLength": 3})
        self.assertEqual(compact["properties"]["payload"], schema["properties"]["payload"])
        self.assertEqual(compact["properties"]["units"], {"allOf": [{"enum": ["us", "si"]}], "default": "us"})
        original_validator = Draft202012Validator(schema)
        compact_validator = Draft202012Validator(compact)
        valid = {"description": "text", "payload": {"description": "Literal value"}}
        for value in (valid, valid | {"units": "si"}, valid | {"description": "x"}, valid | {"units": "kg"}, {}):
            self.assertEqual(original_validator.is_valid(value), compact_validator.is_valid(value))

    def test_explicit_defaults_only_match_omitted_arguments(self):
        schema = {"properties": {"units": {"default": "us"}, "number": {"default": 1}, "anything": True}}
        expected = {"id": "nws/forecast", "args": {"city": "Seattle"}}
        action = {"tool_id": expected["id"], "arguments": expected["args"] | {"units": "us", "number": True,
                                                                           "anything": "arbitrary"}}
        self.assertEqual(normalize_defaults(action, expected, schema)["arguments"],
                         {"city": "Seattle", "number": True, "anything": "arbitrary"})
        required = expected | {"args": expected["args"] | {"units": "si"}}
        self.assertEqual(normalize_defaults(action, required, schema), action)


class SerialPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_speculative_call_is_withheld_and_replanned_after_receipt(self):
        first, second = "provider/lookup", "provider/read"
        actions = [{"tool_id": first, "arguments": {}}, {"tool_id": second, "arguments": {"id": "guess"}}]
        actual = {"tool_id": second, "arguments": {"id": "receipt"}}
        agent = Mock(wraps=NativeModel(1))
        agent.propose.side_effect = [(actions, 10, 5), ([actual], 12, 5)]
        agent.arguments = {}
        cached = {tool_id: Mock(id=tool_id, revision="fixture", tool=Mock(inputSchema={})) for tool_id in (first, second)}
        gateway = Mock(invoke=AsyncMock(return_value=Mock(isError=False, structuredContent={"id": "receipt"})))
        matches = [{"id": tool_id, "inputSchema": {}} for tool_id in (first, second)]
        case = {"id": "fixture", "request": "Read the record returned by lookup.",
                "calls": [{"id": first, "args": {}}, {"id": second, "args": actual["arguments"]}]}
        row = await evaluate(agent, None, gateway, cached, matches, "all_serial", case, 17)
        self.assertTrue(row["success"])
        self.assertEqual(row["proposals_withheld"], 1)
        self.assertEqual(gateway.invoke.await_args.args, (second, "fixture", {"id": "receipt"}))
        self.assertEqual(gateway.invoke.await_count, 2)
