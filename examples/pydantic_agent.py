from pydantic_ai import Agent

from examples.local_tools import find_accessible_entry, list_exhibits


def build_agent(model, selector, query):
    functions = {fn.__name__: fn for fn in (list_exhibits, find_accessible_entry)}
    hits = selector.rank(query, k=1)
    return Agent(model, tools=[functions[hit.tool["name"]] for hit in hits])


async def run_gateway(args):
    from pydantic_ai.messages import (ModelRequest, ModelResponse, SystemPromptPart, TextPart, ToolCallPart,
                                     ToolReturnPart, UserPromptPart)
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.tools import ToolDefinition

    from examples.gateway_agent import Proposal, SYSTEM, Turn, gateway_session, run_workflow

    model = OpenAIChatModel(args.model, provider=OpenAIProvider(base_url=args.url, api_key="local"))

    async def advance(query, observations, menu):
        messages = [ModelRequest(parts=[SystemPromptPart(SYSTEM), UserPromptPart(query)])]
        for entry in observations:
            call = entry.proposal
            messages.extend((ModelResponse(parts=[ToolCallPart(call.name, call.args, call.id)]),
                             ModelRequest(parts=[ToolReturnPart(call.name, entry.content, call.id)])))
        parameters = ModelRequestParameters(function_tools=[
            ToolDefinition(name=tool["name"], description=tool["description"],
                           parameters_json_schema=tool["parameters"]) for tool in menu
        ])
        response = await model.request(messages, {"temperature": 0, "parallel_tool_calls": False}, parameters)
        proposals = tuple(Proposal(part.tool_name, part.args_as_dict(), part.tool_call_id)
                          for part in response.parts if isinstance(part, ToolCallPart))
        return Turn("".join(part.content for part in response.parts if isinstance(part, TextPart)), proposals)

    async with gateway_session(args) as client:
        return await run_workflow(client, args.query, advance)


if __name__ == "__main__":
    import asyncio

    from examples.gateway_agent import arguments

    print(asyncio.run(run_gateway(arguments())).answer)
