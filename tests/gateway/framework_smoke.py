"""Verify both framework demos using a real local model and two real MCP servers."""
import asyncio
from datetime import datetime
import json
import re

from examples.gateway_agent import arguments
from examples.langchain_agent import run_gateway as langchain
from examples.pydantic_agent import run_gateway as pydantic


async def smoke(args):
    records = []
    for name, run in (("LangChain", langchain), ("Pydantic AI", pydantic)):
        for trial in (1, 2):
            workflow = await run(args)
            order = [entry.tool_id for entry in workflow.observations]
            if order != ["clock/get_current_time", "time_math/minutes_since_midnight"]:
                raise ValueError(f"{name} did not complete both gateway calls in order")
            clock, calculation = workflow.observations
            timestamp = clock.content["datetime"]
            if clock.content["timezone"] != "Asia/Tokyo" or calculation.proposal.args != {"timestamp": timestamp}:
                raise ValueError(f"{name} did not pass the exact clock receipt")
            instant = datetime.fromisoformat(timestamp)
            minutes = instant.hour * 60 + instant.minute
            if calculation.content["minutes"] != minutes:
                raise ValueError(f"{name} received incorrect time calculation")
            if timestamp not in workflow.answer or not re.search(rf"\b{minutes}\b", workflow.answer):
                raise ValueError(f"{name} did not report the verified timestamp and minute count")
            records.append({"framework": name, "trial": trial, "tool_order": order,
                            "timestamp_matches_receipt": True, "calculation_correct": True,
                            "answer_matches_receipts": True, "withheld_proposals": workflow.withheld_proposals})
    return records


if __name__ == "__main__":
    print(json.dumps(asyncio.run(smoke(arguments())), indent=2))
