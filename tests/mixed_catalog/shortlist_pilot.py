"""Benchmark two ways to reduce cold hybrid indexing on pinned catalogs."""
import argparse
import asyncio
from hashlib import file_digest
import json
from math import log2
from pathlib import Path
import platform
import random
import resource
from statistics import median, quantiles
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

from gateway_profile import etom_payload

PROTOCOL = Path(__file__).with_name("shortlist_protocol.json")
API = Path("data/onnx_int8_holdout.json")
STRATEGIES = ("bm25", "hybrid", "lexical", "bucketed")


def digest(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def api_payload():
    data = json.loads(API.read_text())
    ids = {key: f"api/{tool['name']}" for key, tool in data["tools"].items()}
    entries = [
        {"server": "api", "tool": {"name": tool["name"], "description": tool["description"],
                                   "inputSchema": {"type": "object"}},
         "invocable": True, "revision": "static-fixture"}
        for tool in data["tools"].values()
    ]
    cases = [{"id": row["id"], "query": row["query"], "family": row["family"],
              "expected": [ids[key] for key in row["positives"]]} for row in data["queries"]]
    return {"tools": entries, "cases": cases}


def percentile(values):
    return {"p50": median(values), "p95": quantiles(values, n=100)[94]}


async def worker(args, protocol):
    started = perf_counter()
    import numpy as np
    from mcp.types import Tool
    from toolrank_cpu.catalog_lexical import CatalogLexicalIndex, top_indices
    from toolrank_cpu.gateway import Entry, Gateway, GatewayConfig, SearchIndex

    import_ms = 1000 * (perf_counter() - started)

    class LexicalShortlist(SearchIndex):
        def __init__(self, encoder):
            super().__init__(encoder)
            self.full = SearchIndex(encoder)
            self.fallbacks = 0

        def search(self, entries, query, limit, min_cosine, *, invocable_only=False):
            if min_cosine is not None or invocable_only:
                raise ValueError("Pilot accepts only full-catalog uncalibrated search")
            texts = [entry.text for entry in entries]
            if texts != self.key:
                self.lexical = CatalogLexicalIndex(texts)
                self.key = texts
                self.vectors = {}
            tfidf, bm25 = self.lexical.scores(query)
            lexical = 0.4 * tfidf + 0.15 * bm25
            positive = np.flatnonzero(lexical > 0)
            if len(positive) < limit:
                self.fallbacks += 1
                self.full.vectors = self.vectors
                matches = self.full.search(entries, query, limit, None)
                self.vectors = self.full.vectors
                return matches
            indices = np.sort(positive[top_indices(lexical[positive], protocol["candidate_width"])])
            candidates = [texts[i] for i in indices]
            missing = list(dict.fromkeys(text for text in candidates if text not in self.vectors))
            for start in range(0, len(missing), 32):
                batch = missing[start:start + 32]
                self.vectors.update(zip(batch, self.encoder.encode(batch), strict=True))
            documents = np.stack([self.vectors[text] for text in candidates])
            query_vector = self.encoder.encode([self.encoder.prefix + query])[0]
            cosine = np.einsum("ij,j->i", documents, query_vector)
            scores = cosine + 0.4 * tfidf[indices] + 0.15 * bm25[indices]
            return [entries[indices[i]].public() | {"cosine": float(cosine[i]),
                                                     "ranking_score": float(scores[i])}
                    for i in top_indices(scores, limit)]

    class BucketedIndex(SearchIndex):
        def _update_embeddings(self, texts):
            super()._update_embeddings(sorted(texts, key=len))
            self.documents = np.stack([self.vectors[text] for text in texts])

    payload = json.loads(args.fixture.read_text())
    entries = [Entry(row["server"], Tool.model_validate(row["tool"]), row["invocable"], row["revision"])
               for row in payload["tools"]]
    config = GatewayConfig.model_validate({"ranking": "bm25" if args.worker == "bm25" else "hybrid",
                                           "servers": [{"id": name, "command": sys.executable, "discover": []}
                                                       for name in sorted({entry.server for entry in entries})]})
    started = perf_counter()
    gateway = Gateway.from_config(config, args.checkpoint if args.worker != "bm25" else None)
    if args.worker == "lexical":
        gateway.index = LexicalShortlist(gateway.index.encoder)
    elif args.worker == "bucketed":
        gateway.index = BucketedIndex(gateway.index.encoder)
    setup_ms = 1000 * (perf_counter() - started)
    for entry in entries:
        upstream = gateway.upstreams[entry.server]
        upstream.entries[entry.id] = entry
        upstream.state = "ready"
    cases = payload["cases"]
    shuffled = random.Random(args.seed).sample(cases, len(cases))
    observations = {}
    times = []
    for case in shuffled:
        started = perf_counter()
        result = await gateway.search(case["query"], protocol["return_limit"])
        times.append(1000 * (perf_counter() - started))
        observations[case["id"]] = [match["id"] for match in result["matches"]]
        if len(times) == 1:
            first_vectors = len(gateway.index.vectors)
            first_fallback = getattr(gateway.index, "fallbacks", 0)
    return {"rankings": observations, "run": {
        "seed": args.seed, "first_case": shuffled[0]["id"],
        "first_documents": first_vectors, "first_used_fallback": bool(first_fallback),
        "cold_first_ms": import_ms + setup_ms + times[0],
        "after_first_search_ms": percentile(times[1:]),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "fallback_queries": getattr(gateway.index, "fallbacks", 0),
        "final_documents": len(gateway.index.vectors)}}


def scores(ranks, cases):
    ndcg = []
    hit = []
    complete = []
    accepted = []
    for case in cases:
        expected = set(case["expected"])
        order = ranks[case["id"]]
        if expected:
            present = set(order) & expected
            gain = sum(1 / log2(i + 2) for i, key in enumerate(order) if key in expected)
            ndcg.append(gain / sum(1 / log2(i + 2) for i in range(min(10, len(expected)))))
            hit.append(bool(present))
            complete.append(present == expected)
        else:
            accepted.append(bool(order))
    return ndcg, {"ndcg_at_10": sum(ndcg) / len(ndcg), "hit_at_10": sum(hit) / len(hit),
                  "complete_at_10": sum(complete) / len(complete), "supported": len(ndcg),
                  "unsupported": len(accepted), "unsupported_accepted": sum(accepted)}


def run(args, protocol):
    from toolrank.baseline_comparison import paired_interval
    from toolrank_cpu.rank_onnx import checkpoint_checksums

    metadata = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "checkpoint": checkpoint_checksums(args.checkpoint, "embedding-int8.onnx"),
                "hardware": {"cpu": next(line.split(":", 1)[1].strip()
                                         for line in Path("/proc/cpuinfo").read_text().splitlines()
                                         if line.startswith("model name")),
                             "platform": platform.platform(), "python": platform.python_version()},
                "protocol_sha256": digest(PROTOCOL), "source_sha256": digest(__file__),
                "protocol": protocol, "datasets": {}, "external_compute_usd": 0}
    with TemporaryDirectory() as directory:
        fixture = Path(directory) / "catalog.json"
        for dataset in args.datasets:
            if dataset == "api":
                payload, source = api_payload(), {str(API): digest(API)}
            else:
                payload, source = etom_payload(args.etom)
                for case in payload["cases"]:
                    case["family"] = min(tool.partition("/")[0] for tool in case["expected"])
            fixture.write_text(json.dumps(payload))
            observations = {}
            for strategy in STRATEGIES:
                workers = []
                for seed in protocol["seeds"]:
                    command = [sys.executable, __file__, "--fixture", str(fixture), "--worker", strategy,
                               "--checkpoint", str(args.checkpoint), "--seed", str(seed)]
                    workers.append(json.loads(subprocess.check_output(command, timeout=240)))
                observations[strategy] = workers
                print(dataset, strategy, [round(worker["run"]["cold_first_ms"]) for worker in workers],
                      flush=True)
            baseline = observations["hybrid"][0]["rankings"]
            baseline_scores, _ = scores(baseline, payload["cases"])
            groups = [case["family"] for case in payload["cases"] if case["expected"]]
            rows = {}
            for strategy, workers in observations.items():
                runs = []
                for worker in workers:
                    order = worker["rankings"]
                    per_query, result = scores(order, payload["cases"])
                    runs.append(worker["run"] | {"quality": result,
                        "exact_top10_fraction": sum(row == baseline[key] for key, row in order.items()) / len(order),
                        "paired_ndcg": paired_interval(per_query, baseline_scores, groups)})
                rows[strategy] = runs
            metadata["datasets"][dataset] = {"source": source, "tools": len(payload["tools"]), "runs": rows}
            args.output.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--etom", type=Path, default=Path(".work/next/etom"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=("api", "etom-plain"), default=("api", "etom-plain"))
    parser.add_argument("--worker", choices=STRATEGIES, help=argparse.SUPPRESS)
    parser.add_argument("--fixture", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if args.worker:
        print(json.dumps(asyncio.run(worker(args, protocol))))
    else:
        if not args.output:
            parser.error("--output is required")
        run(args, protocol)
