import argparse
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from examples.local_tools import CATALOG, QUERY, find_accessible_entry, list_exhibits
from toolrank.selector import ToolSelector


def build_agent(chat_model, selector, query):
    functions = {fn.__name__: fn for fn in (list_exhibits, find_accessible_entry)}
    hits = selector.rank(query, k=1)
    return create_agent(chat_model, tools=[functions[hit.tool["name"]] for hit in hits])


class LocalToolCallingModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    selector = ToolSelector.from_checkpoint(args.checkpoint, CATALOG)
    model = LocalToolCallingModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "find_accessible_entry", "args": {}, "id": "demo"}]),
        AIMessage(content="Use the north entrance."),
    ]))
    result = build_agent(model, selector, QUERY).invoke({"messages": [{"role": "user", "content": QUERY}]})
    print([message.content for message in result["messages"] if message.type == "tool"])
