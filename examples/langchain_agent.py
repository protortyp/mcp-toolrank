import json

from langchain.agents import create_agent

from examples.local_tools import find_accessible_entry, list_exhibits


def build_agent(chat_model, selector, query):
    functions = {fn.__name__: fn for fn in (list_exhibits, find_accessible_entry)}
    hits = selector.rank(query, k=1)
    return create_agent(chat_model, tools=[functions[hit.tool["name"]] for hit in hits])


async def run_gateway(args):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_openai import ChatOpenAI
    from langsmith import tracing_context

    from examples.gateway_agent import Proposal, SYSTEM, Turn, gateway_session, run_workflow

    model = ChatOpenAI(model=args.model, base_url=args.url, api_key="local", temperature=0)

    async def advance(query, observations, menu):
        messages = [SystemMessage(content=SYSTEM), HumanMessage(content=query)]
        for entry in observations:
            call = entry.proposal
            messages.extend((AIMessage(content="", tool_calls=[{"name": call.name, "args": call.args, "id": call.id}]),
                             ToolMessage(content=json.dumps(entry.content), tool_call_id=call.id)))
        model_with_tools = model.bind_tools([{"type": "function", "function": tool} for tool in menu],
                                            parallel_tool_calls=False) if menu else model
        response = await model_with_tools.ainvoke(messages)
        if response.invalid_tool_calls:
            raise RuntimeError("Model proposed malformed tool calls")
        return Turn(response.text, tuple(Proposal(call["name"], call["args"], call["id"])
                                         for call in response.tool_calls))

    with tracing_context(enabled=False):
        async with gateway_session(args) as client:
            return await run_workflow(client, args.query, advance)


if __name__ == "__main__":
    import asyncio

    from examples.gateway_agent import arguments

    print(asyncio.run(run_gateway(arguments())).answer)
