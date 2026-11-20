"""Measure gateway shortlists and abstention on a pinned NWS operation catalog."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.openapi_catalog import catalog_from_openapi
from examples.rank_onnx import OnnxToolRank, calibrate, checkpoint_checksums, top_indices


HERE = Path(__file__).parent
FEATURES = ("cosine", "hybrid_max", "margin", "standardized")


def fingerprint(path):
    return sha256(path.read_bytes()).hexdigest()


def read_requests(path, names):
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    used = set()
    for row in rows:
        row["expected"] = None if row["operation"] == "-" else "weather__" + row["operation"]
        if row["split"] not in ("calibration", "holdout") or row["id"] in used:
            raise ValueError("invalid split or duplicate request")
        if row["expected"] and row["expected"] not in names:
            raise ValueError(f"unknown operation for {row['id']}")
        used.add(row["id"])
    return rows


def confidence(cosine, hybrid):
    second, first = np.partition(cosine, -2)[-2:]
    return {"cosine": float(first), "hybrid_max": float(hybrid.max()),
            "margin": float(first - second),
            "standardized": float((first - cosine.mean()) / max(cosine.std(), 1e-6))}


def score(ranker, rows):
    data = []
    for row in rows:
        start = perf_counter()
        cosine = ranker.cosine_scores(row["query"])
        tfidf, bm25 = ranker.lexical_index.scores(row["query"])
        hybrid = cosine + .4 * tfidf + .15 * bm25
        data.append(({**row, "features": confidence(cosine, hybrid),
                      "seconds": perf_counter() - start},
                     {"hybrid": hybrid, "dense": cosine, "bm25": bm25}))
    return data


def evaluate(samples, tools, method, signal=None, threshold=None):
    supported = unsupported = accepted = false_accept = hit1 = hit5 = 0
    total_tool_bytes = total_shortlist_bytes = 0
    misses = []
    for row, scores in samples:
        included = signal is None or row["features"][signal] >= threshold
        order = []
        if included:
            eligible = np.arange(len(tools))
            # Production gateway applies its cosine threshold to each candidate.
            if signal == "cosine":
                eligible = eligible[scores["dense"] >= threshold]
            ranking = scores[method][eligible]
            order = eligible[top_indices(ranking, 5)]
        if row["expected"] is None:
            unsupported += 1
            false_accept += bool(len(order))
        else:
            supported += 1
            accepted += bool(len(order))
            names = [tools[index]["name"] for index in order]
            hit1 += row["expected"] in names[:1]
            hit5 += row["expected"] in names
            if row["expected"] not in names:
                misses.append(row["id"])
        total_tool_bytes += json_size(tools)
        total_shortlist_bytes += json_size([tools[index] for index in order])
    return {"supported": supported, "unsupported": unsupported,
            "accepted_supported": accepted, "accepted_unsupported": false_accept,
            "supported_hit_at_1": hit1, "supported_hit_at_5": hit5,
            "missed_supported_ids": misses,
            "tool_json_bytes": total_tool_bytes, "shortlist_json_bytes": total_shortlist_bytes,
            "tool_json_reduction": 1 - total_shortlist_bytes / total_tool_bytes}


def pilot_gate(report):
    unfiltered, filtered = report["hybrid"], report["hybrid_cosine_cutoff"]
    checks = {
        "top_five_recall_at_least_90_percent": unfiltered["supported_hit_at_5"] / unfiltered["supported"] >= .90,
        "tool_json_reduction_at_least_80_percent": unfiltered["tool_json_reduction"] >= .80,
        "false_acceptance_at_most_10_percent": filtered["accepted_unsupported"] / filtered["unsupported"] <= .10,
        "supported_coverage_at_least_85_percent": filtered["accepted_supported"] / filtered["supported"] >= .85,
    }
    return {"passed": all(checks.values()), "checks": checks}


def json_size(value):
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode())


def run(spec, checkpoint):
    protocol = json.loads((HERE / "protocol.json").read_text())
    if fingerprint(spec) != protocol["spec_sha256"]:
        raise ValueError("OpenAPI snapshot hash differs from this pilot")
    tools = catalog_from_openapi(json.loads(spec.read_text()), "weather")
    rows = read_requests(HERE / "requests.tsv", {tool["name"] for tool in tools})
    start = perf_counter()
    ranker = OnnxToolRank(checkpoint, tools, hybrid=True)
    indexing_seconds = perf_counter() - start
    ranker.cosine_scores("Read the weather metadata.")
    development = score(ranker, [row for row in rows if row["split"] == "calibration"])
    cases = [(row["id"], {row["expected"]} if row["expected"] else set()) for row, _ in development]
    by_id = {row["id"]: row["features"] for row, _ in development}
    thresholds = {}
    for signal in FEATURES:
        thresholds[signal] = calibrate(cases, max_false_accept=.1, confidence=.95,
                                       score_requests=lambda ids: [by_id[key][signal] for key in ids])
    holdout = score(ranker, [row for row in rows if row["split"] == "holdout"])
    report = {method: evaluate(holdout, ranker.tools, method) for method in ("hybrid", "dense", "bm25")}
    for signal, threshold in thresholds.items():
        report[f"hybrid_{signal}_cutoff"] = evaluate(holdout, ranker.tools, "hybrid", signal,
                                                       threshold["min_score"])
    return {"catalog_tools": len(tools), "spec_sha256": fingerprint(spec),
            "requests_sha256": fingerprint(HERE / "requests.tsv"),
            "protocol_sha256": fingerprint(HERE / "protocol.json"), "source_sha256": fingerprint(Path(__file__)),
            "checkpoint": checkpoint_checksums(checkpoint, "embedding-int8.onnx"),
            "cpu_threads": 4, "startup_and_index_ms": 1000 * indexing_seconds,
            "warm_query_p95_ms": 1000 * float(np.quantile([row["seconds"] for row, _ in holdout], .95)),
            "calibration": thresholds, "holdout": report, "gate": pilot_gate(report), "external_compute_usd": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.spec, args.checkpoint), indent=2) + "\n")
