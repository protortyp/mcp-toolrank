"""Compare the packaged gateway indexes on matched MCP catalogs."""
import argparse
import asyncio
import json
from math import log2
from pathlib import Path
import random
import resource
from statistics import median, quantiles
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter


def elapsed(start):
    return 1000 * (perf_counter() - start)


async def measure(args):
    start = perf_counter()
    from mcp.types import Tool
    from toolrank_cpu.gateway import Entry, Gateway, GatewayConfig

    load_ms = elapsed(start)
    payload = json.loads(args.fixture.read_text())
    tools = [Entry(row["server"], Tool.model_validate(row["tool"]), row["invocable"], row["revision"])
             for row in payload["tools"]]
    names = sorted({entry.server for entry in tools})
    config = GatewayConfig.model_validate({"ranking": args.worker, "servers": [
        {"id": name, "command": sys.executable, "discover": []} for name in names]})
    start = perf_counter()
    gateway = Gateway.from_config(config, args.checkpoint if args.worker == "hybrid" else None)
    setup_ms = elapsed(start)
    for entry in tools:
        upstream = gateway.upstreams[entry.server]
        upstream.entries[entry.id] = entry
        upstream.state = "ready"
    cases = payload["cases"]
    start = perf_counter()
    await gateway.search(cases[0]["query"], 10)
    first_search_ms = elapsed(start)
    samples = []
    observed = {}
    randomizer = random.Random(args.seed)
    for _ in range(payload["rounds"]):
        for case in randomizer.sample(cases, len(cases)):
            start = perf_counter()
            result = await gateway.search(case["query"], 10)
            samples.append(elapsed(start))
            observed[case["id"]] = {"ids": [row["id"] for row in result["matches"]],
                                    "decision": result["decision"]}
    process = {"seed": args.seed, "import_ms": load_ms, "model_setup_ms": setup_ms,
               "first_search_ms": first_search_ms, "cold_first_search_ms": load_ms + setup_ms + first_search_ms,
               "warm_search_p50_ms": median(samples), "warm_search_p95_ms": quantiles(samples, n=100)[94],
               "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               "model_libraries": sorted({"onnxruntime", "tokenizers"} & sys.modules.keys())}
    return {"process": process, "observed": observed}


def mixed_payload():
    import csv
    from run import CASES, INDEPENDENT, NWS, catalog, digest

    tools = catalog()
    names = {entry.tool.name: entry.id for entry in tools}
    if len(names) != len(tools):
        raise ValueError("This fixture identifies tools by unique operation name")
    with CASES.open(newline="", encoding="utf-8") as stream:
        cases = [{"id": row["id"], "query": f"{row['intent_a']} {row['intent_b']}",
                  "expected": sorted(names[name] for name in {row["tool_a"], row["tool_b"]} - {"-"})}
                 for row in csv.DictReader(stream, delimiter="\t")]
    payload = {"tools": [{"server": row.server, "tool": row.tool.model_dump(),
                          "invocable": row.invocable, "revision": row.revision} for row in tools], "cases": cases}
    return payload | {"rounds": 3}, {path.name: digest(path) for path in (CASES, INDEPENDENT, NWS)}


def etom_payload(root):
    from toolrank.etom import FILES, load

    _, labels, audit = load("plain", root)
    registry = json.loads((root / FILES[0]).read_text())
    providers = sorted(registry["servers"], key=lambda server: server["server_name"])
    ids = {}
    entries = []
    for index, server in enumerate(providers):
        alias = f"provider-{index:04d}"
        for operation in server["tools"]:
            name = operation["tool_name"]
            ids[f"{server['server_name']}::{name}"] = f"{alias}/{name}"
            entries.append({"server": alias, "invocable": False, "revision": "static-fixture", "tool": {
                "name": name, "description": operation["tool_description"], "inputSchema": operation["input_schema"],
            }})
    cases = [{"id": id_, "query": query, "expected": sorted(ids[positive] for positive in positives)}
             for id_, query, _, positives in labels]
    return {"cases": cases, "tools": entries, "rounds": 1}, audit


def quality(observed, cases):
    counts = {"full_positive_retained": 0, "required_tool_hits": 0,
              "unsupported_with_suggestions": 0, "unsupported_no_lexical_overlap": 0}
    ndcg = []
    for case in cases:
        relevant = set(case["expected"])
        actual = observed[case["id"]]
        if relevant:
            counts["full_positive_retained"] += relevant.issubset(actual["ids"])
            counts["required_tool_hits"] += len(relevant.intersection(actual["ids"]))
            discounts = [1 / log2(i + 2) for i in range(10)]
            dcg = sum(discount for discount, id_ in zip(discounts, actual["ids"]) if id_ in relevant)
            ndcg.append(dcg / sum(discounts[:len(relevant)]))
        else:
            counts["unsupported_with_suggestions"] += bool(actual["ids"])
            counts["unsupported_no_lexical_overlap"] += actual["decision"] == "no_lexical_overlap"
    return counts | {"ndcg_at_10": sum(ndcg) / len(ndcg)}


def run(args):
    import platform
    from toolrank_cpu.rank_onnx import checkpoint_checksums
    from run import digest

    payload, data = etom_payload(args.etom) if args.etom else mixed_payload()
    cases = payload["cases"]
    seeds = (11, 23, 37)
    results = {}
    with TemporaryDirectory() as temporary:
        fixture = Path(temporary) / "catalog.json"
        fixture.write_text(json.dumps(payload))
        for strategy in ("bm25", "hybrid"):
            workers = []
            for seed in seeds:
                command = [sys.executable, __file__, "--worker", strategy, "--fixture", str(fixture),
                           "--checkpoint", str(args.checkpoint), "--seed", str(seed)]
                workers.append(json.loads(subprocess.check_output(command, text=True, timeout=180)))
            observed = workers[0]["observed"]
            if any(worker["observed"] != observed for worker in workers):
                raise RuntimeError(f"Nonreproducible gateway ranking: {strategy}")
            processes = [worker["process"] for worker in workers]
            metrics = {key: median(row[key] for row in processes)
                       for key in processes[0] if key not in {"seed", "model_libraries"}}
            results[strategy] = quality(observed, cases) | {"process_medians": metrics, "processes": processes}
            if args.etom is None:
                results[strategy]["results"] = observed
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    output = {"hardware": {"cpu": cpu, "platform": platform.platform(), "python": platform.python_version()},
              "scope": "Fresh processes with warmed filesystem. Timings exclude transport, disk download, fixture "
                       "loading and Python startup. First search includes index construction. Warm latency uses the "
                       "same shuffled cases. Peak RSS is process-wide on Linux.",
              "dataset": data, "limit": 10, "catalog_tools": len(payload["tools"]),
              "supported_requests": sum(bool(row["expected"]) for row in cases),
              "unsupported_requests": sum(not row["expected"] for row in cases), "seeds": list(seeds),
              "warm_rounds": payload["rounds"], "source_sha256": digest(Path(__file__)),
              "checkpoint_checksums": checkpoint_checksums(args.checkpoint, "embedding-int8.onnx"),
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "external_compute_usd": 0, "results": results}
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--etom", type=Path, help="use a local pinned ETOM snapshot instead of the mixed catalog")
    parser.add_argument("--worker", choices=("bm25", "hybrid"), help=argparse.SUPPRESS)
    parser.add_argument("--fixture", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(asyncio.run(measure(args))))
    else:
        if args.output is None:
            parser.error("--output is required")
        run(args)
