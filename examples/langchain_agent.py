from langchain.agents import create_agent

from examples.local_tools import find_accessible_entry, list_exhibits


def build_agent(chat_model, selector, query):
    functions = {fn.__name__: fn for fn in (list_exhibits, find_accessible_entry)}
    hits = selector.rank(query, k=1)
    return create_agent(chat_model, tools=[functions[hit.tool["name"]] for hit in hits])


async def run_gateway(args):
    from langchain_core.tools import StructuredTool
    from langchain_openai import ChatOpenAI

    from examples.gateway_clock import clock_tool

    model = ChatOpenAI(model=args.model, base_url=args.url, api_key="local", temperature=0)
    async with clock_tool(args) as clock:
        tool = StructuredTool.from_function(coroutine=clock)
        agent = create_agent(model, tools=[tool], system_prompt="Call the clock tool to answer time queries.")
        response = await agent.ainvoke({"messages": [{"role": "user", "content": args.query}]})
        print(response["messages"][-1].content)


if __name__ == "__main__":
    import asyncio

    from examples.gateway_clock import arguments

    asyncio.run(run_gateway(arguments()))
