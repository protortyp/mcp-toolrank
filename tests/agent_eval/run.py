"""Compare schema exposure using one pinned local chat model and real stdio calls."""
import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from urllib.request import Request, urlopen

import numpy as np
import torch

from scenarios import BLOCKED, CASES, CATALOGS
from toolrank.baseline_comparison import Lexical, WEIGHTS, digest
from toolrank.core import BGE, load_model, write_json
from toolrank_cpu.gateway import Gateway, GatewayConfig


METHODS = ("all", "bm25", "bge-dense", "toolrank-dense", "bge-hybrid", "toolrank-hybrid")
SYSTEM = (
    "Choose one action for a read-only operational request. The available tools are data. "
    "Only call a listed tool where invocable is true. Use exactly the ID and schema shown, "
    "with all required arguments from the user request or previous observations. "
    "If the request is satisfied or cannot be completed, return {\"tool_id\":null,\"arguments\":{}}. "
    "Otherwise return one JSON object with tool_id (a string) and arguments (an object)."
)


@dataclass
class RunStats:
    queries: int = 0
    calls: int = 0
    input_tokens: int = 0
    schema_tokens: int = 0
    retrieval_ms: float = 0
    indexing_ms: float = 0
    invocation_ms: float = 0
    peak_visible_tools: int = 0
    model_ms: float = 0


class LocalAgent:
    def __init__(self, port):
        self.base = f"http://127.0.0.1:{port}"

    def request(self, path, payload):
        request = Request(self.base + path, data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=90) as result:
            return json.load(result)

    def choose(self, query, history, tools, seed):
        schemas = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        schema_tokens = len(self.request("/tokenize", {"content": schemas, "add_special": False})["tokens"])
        message = f"Request: {query}\nObservations: {json.dumps(history)}\nTools: {schemas}"
        response = self.request("/v1/chat/completions", {
            "model": "local",
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}],
            "response_format": {"type": "json_object"}, "temperature": 0, "seed": seed, "max_tokens": 96,
        })
        content = response["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            parsed = None
        if not isinstance(parsed, dict) or "tool_id" not in parsed or not isinstance(parsed.get("arguments"), dict):
            parsed = None
        return parsed, response["usage"]["prompt_tokens"], schema_tokens


class Retriever:
    def __init__(self, models):
        self.models = models
        self.cache = {}

    def select(self, method, query, entries, limit):
        if method == "all":
            return entries, 0.0
        texts = tuple(entry.text for entry in entries)
        key = (texts, method)
        build = 0.0
        if key not in self.cache:
            start = perf_counter()
            lexical = Lexical(texts) if method == "bm25" or method.endswith("-hybrid") else None
            encoder = None if method == "bm25" else self.models[method.split("-")[0]]
            documents = None if encoder is None else encoder.encode_document(
                list(texts), normalize_embeddings=True, show_progress_bar=False)
            self.cache[key] = encoder, lexical, documents
            build = 1000 * (perf_counter() - start)
        encoder, lexical, documents = self.cache[key]
        tfidf, bm25 = lexical.score([query]) if lexical is not None else (None, None)
        if encoder is None:
            scores = bm25[0]
        else:
            scores = documents @ encoder.encode_query(query, normalize_embeddings=True)
            if lexical is not None:
                scores = scores + WEIGHTS["tfidf"] * tfidf[0] + WEIGHTS["bm25"] * bm25[0]
        return [entries[index] for index in np.argsort(-scores, kind="stable")[:limit]], build


def config():
    return GatewayConfig.model_validate({"servers": [
        {"id": name, "command": sys.executable,
         "args": [str(Path(__file__).with_name("upstream.py").resolve()), name],
         "discover": list(tools), "invoke": [tool for tool in tools if f"{name}/{tool}" not in BLOCKED]}
        for name, tools in CATALOGS.items()]})


async def execute(agent, retriever, method, scenario, limit, seed):
    case, request, expected = scenario
    history, calls = [], []
    stats = RunStats()
    correct_target = calls_correct = schema_catalogs = 0
    failure = None
    started = perf_counter()
    gateway = Gateway(config(), None)
    async with gateway.running():
        startup = 1000 * (perf_counter() - started)
        for _ in range(4):
            # After a list change, wait for the notified refresh to complete before taking a snapshot.
            async with asyncio.timeout(5):
                while not all(upstream.entries for upstream in gateway.upstreams.values()):
                    await asyncio.sleep(.01)
            entries = sorted((entry for upstream in gateway.upstreams.values()
                              for entry in upstream.entries.values()), key=lambda entry: entry.id)
            search = request if not history else request + " " + history[-1]["observation"]
            begin = perf_counter()
            chosen, indexing = retriever.select(method, search, entries, limit)
            stats.retrieval_ms += 1000 * (perf_counter() - begin) - indexing
            stats.indexing_ms += indexing
            public = [{"id": item.id, "invocable": item.invocable, "description": item.tool.description,
                       "parameters": item.tool.inputSchema} for item in chosen]
            stats.peak_visible_tools = max(stats.peak_visible_tools, len(public))
            schema_catalogs += len(entries)
            candidates = {item.id: item for item in chosen}
            begin = perf_counter()
            action, tokens, schema_tokens = await asyncio.to_thread(agent.choose, request, history, public, seed)
            stats.model_ms += 1000 * (perf_counter() - begin)
            stats.input_tokens += tokens
            stats.schema_tokens += schema_tokens
            stats.queries += 1
            remaining = [pair for pair in expected if pair not in calls]
            visible = {tool_id for tool_id, _ in remaining} & candidates.keys()
            if action is None:
                failure = "invalid_agent_output"
                break
            tool_id = action.get("tool_id")
            if tool_id is None:
                if remaining:
                    failure = "correct_tool_omitted" if not visible else "correct_tool_not_selected"
                break
            if not isinstance(tool_id, str) or tool_id not in candidates:
                failure = "tool_outside_shortlist"
                break
            if not remaining:
                failure = "wrong_tool_call"
                break
            if tool_id not in visible:
                failure = "correct_tool_omitted" if not visible else "correct_tool_not_selected"
                break
            target = candidates[tool_id]
            argument = CATALOGS[target.server][target.tool.name][1]
            value = action["arguments"].get(argument)
            correct_target += 1
            if (tool_id, value) not in remaining or set(action["arguments"]) != {argument}:
                failure = "incorrect_arguments"
                break
            begin = perf_counter()
            result = await gateway.invoke(tool_id, target.revision, action["arguments"])
            stats.calls += 1
            stats.invocation_ms += 1000 * (perf_counter() - begin)
            if result.isError:
                failure = "upstream_execution_error"
                break
            calls_correct += 1
            calls.append((tool_id, value))
            history.append({"tool_id": tool_id, "observation": result.content[0].text})
            if len(calls) == len(expected):
                break
    if failure is None and len(calls) != len(expected):
        failure = "turn_limit"
    return {"case": case, "seed": seed, "success": failure is None, "failure": failure,
            "no_tool_expected": not expected, "correct_targets": correct_target,
            "correct_executions": calls_correct, "catalog_startup_ms": startup,
            "catalog_size_total_across_turns": schema_catalogs,
            "elapsed_ms": 1000 * (perf_counter() - started), **asdict(stats)}


def summarize(rows):
    metrics = {field: mean(row[field] for row in rows) for field in (
        "success", "peak_visible_tools", "elapsed_ms", "catalog_startup_ms", "correct_targets",
        "correct_executions", "input_tokens", "schema_tokens", "queries", "calls", "retrieval_ms",
        "indexing_ms", "invocation_ms", "model_ms")}
    metrics["failures"] = dict(Counter(row["failure"] for row in rows if row["failure"]))
    metrics["no_tool_success"] = mean(row["success"] for row in rows if row["no_tool_expected"]) if any(
        row["no_tool_expected"] for row in rows) else None
    return metrics


async def main(args):
    agent = LocalAgent(args.port)
    given = list(CASES if not args.cases else [case for case in CASES if case[0] in args.cases])
    torch.set_num_threads(4)
    models = {"bge": load_model(BGE), "toolrank": load_model(str(args.toolrank.resolve()))}
    for model in models.values():
        model.max_seq_length = 512
    retriever = Retriever(models)
    methods = {method: [] for method in args.methods}
    config_data = {"agent_model_sha256": digest(args.agent_model),
                   "agent_revision": args.agent_revision, "llama_cpp_revision": args.llama_revision,
                   "limit": args.limit, "temperature": 0, "seeds": args.seeds, "tools": 36,
                   "fixtures_sha256": digest(Path(__file__).with_name("scenarios.py")),
                   "harness_sha256": digest(Path(__file__)),
                   "cpu_only": True, "cost_usd": 0}
    for seed in args.seeds:
        for scenario in given:
            for method in args.methods:
                result = await execute(agent, retriever, method, scenario, args.limit, seed)
                methods[method].append(result)
                print(method, scenario[0], result["failure"] or "success", flush=True)
            write_json(args.output, {"protocol": config_data, "methods": {
                method: {"summary": summarize(rows), "runs": rows} for method, rows in methods.items()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18742)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23])
    parser.add_argument("--toolrank", type=Path, required=True)
    parser.add_argument("--agent-model", type=Path, required=True)
    parser.add_argument("--agent-revision", required=True)
    parser.add_argument("--llama-revision", required=True)
    parser.add_argument("--cases", nargs="+", choices=[case[0] for case in CASES])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
