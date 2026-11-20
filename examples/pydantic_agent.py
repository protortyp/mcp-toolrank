import argparse
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from examples.local_tools import CATALOG, QUERY, find_accessible_entry, list_exhibits
from toolrank.selector import ToolSelector


def build_agent(model, selector, query):
    functions = {fn.__name__: fn for fn in (list_exhibits, find_accessible_entry)}
    hits = selector.rank(query, k=1)
    return Agent(model, tools=[functions[hit.tool["name"]] for hit in hits])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    selector = ToolSelector.from_checkpoint(args.checkpoint, CATALOG)
    print([hit.tool["name"] for hit in selector.rank(QUERY, k=1)])
    print(build_agent(TestModel(), selector, QUERY).run_sync(QUERY).output)
