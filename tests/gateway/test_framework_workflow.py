from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from examples.gateway_agent import Proposal, Turn, function_name, run_workflow, tool_menu


def match(id_, *, invocable=True):
    return {"id": id_, "description": "Read a record", "inputSchema": {"type": "object"},
            "revision": "fixture-revision", "invocable": invocable}


def result(content, *, error=False):
    return SimpleNamespace(structuredContent=content, isError=error, content=[])


class FrameworkWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def test_function_aliases_preserve_identity_and_schema(self):
        tools = [match("one/place/name"), match("one_place/name"), match("restricted/read", invocable=False)]
        targets, functions = tool_menu(tools)
        self.assertEqual({row["id"] for row in targets.values()}, {"one/place/name", "one_place/name"})
        self.assertEqual(len(functions), 2)
        self.assertEqual(functions[0]["parameters"], tools[0]["inputSchema"])
        self.assertRegex(function_name("single/" + "long" * 32), r"^[A-Za-z0-9_-]{1,64}$")

    async def test_retains_only_first_proposal_and_searches_after_receipt(self):
        first, second = match("one/read"), match("two/read")
        client = SimpleNamespace(call_tool=AsyncMock(side_effect=[
            result({"matches": [first, second]}), result({"record": "actual"}),
            result({"matches": [second]}), result({"received": "actual"}), result({"matches": []}),
        ]))
        initial = Proposal(function_name(first["id"]), {}, "a")
        speculative = Proposal(function_name(second["id"]), {"record": "guess"}, "b")
        grounded = Proposal(function_name(second["id"]), {"record": "actual"}, "c")
        model = AsyncMock(side_effect=[Turn("", (initial, speculative)), Turn("", (grounded,)), Turn("Done", ())])
        workflow = await run_workflow(client, "Find the record", model)
        self.assertEqual(workflow.withheld_proposals, 1)
        self.assertEqual([entry.tool_id for entry in workflow.observations], [first["id"], second["id"]])
        self.assertEqual(model.await_args_list[1].args[1][0].content, {"record": "actual"})
        self.assertIn('"actual"', client.call_tool.await_args_list[2].args[1]["query"])
        calls = [arguments for name, arguments in [call.args for call in client.call_tool.await_args_list]
                 if name == "invoke_tool"]
        self.assertEqual(calls[1], {"tool_id": second["id"], "revision": second["revision"],
                                    "arguments": {"record": "actual"}})
        self.assertEqual(len(calls), 2)

    async def test_refuses_proposal_outside_current_allowlist(self):
        hidden = match("restricted/read", invocable=False)
        client = SimpleNamespace(call_tool=AsyncMock(return_value=result({"matches": [hidden]})))
        model = AsyncMock(return_value=Turn("", (Proposal(function_name(hidden["id"]), {}, "guess"),)))
        with self.assertRaisesRegex(RuntimeError, "outside the current invocation shortlist"):
            await run_workflow(client, "Find the record", model)
        client.call_tool.assert_awaited_once_with("search_tools", {
            "query": "Find the record", "limit": 5, "invocable_only": True,
        })

    async def test_stale_calibration_is_a_configuration_error(self):
        client = SimpleNamespace(call_tool=AsyncMock(return_value=result({
            "matches": [], "decision": "calibration_scope_mismatch",
        })))
        model = AsyncMock()
        with self.assertRaisesRegex(RuntimeError, "calibration does not match"):
            await run_workflow(client, "Find the record", model)
        model.assert_not_awaited()
