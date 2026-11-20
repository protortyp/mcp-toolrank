from pydantic_ai import Agent

from examples.local_tools import find_accessible_entry, list_exhibits


def build_agent(model, selector, query):
    functions = {fn.__name__: fn for fn in (list_exhibits, find_accessible_entry)}
    hits = selector.rank(query, k=1)
    return Agent(model, tools=[functions[hit.tool["name"]] for hit in hits])


async def run_gateway(args):
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from examples.gateway_clock import clock_tool

    model = OpenAIChatModel(args.model, provider=OpenAIProvider(base_url=args.url, api_key="local"))
    async with clock_tool(args) as clock:
        agent = Agent(model, tools=[clock], instructions="Call the clock tool to answer time queries.")
        message = await agent.run(args.query, model_settings={"temperature": 0})
        print(message.output)


if __name__ == "__main__":
    import asyncio

    from examples.gateway_clock import arguments

    asyncio.run(run_gateway(arguments()))
