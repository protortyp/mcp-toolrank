"""Evaluate the frozen five-candidate verifier on fresh authored NWS requests."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from evaluate import HERE, evaluate, fingerprint, read_requests, score
from examples.openapi_catalog import catalog_from_openapi
from examples.rank_onnx import OnnxToolRank, calibrate, checkpoint_checksums, top_indices
from verifier import Verifier


REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"


def labels(rows):
    return [(row["id"], {row["expected"]} if row["expected"] else set()) for row in rows]


def shortlist(ranker, rows):
    scored = score(ranker, rows)
    choices = [top_indices(values["hybrid"], 5) for _, values in scored]
    return scored, choices


def model_scores(verifier, ranker, rows, choices):
    pairs = [(row["query"], ranker.tools[index]["description"])
             for row, indices in zip(rows, choices, strict=True) for index in indices]
    values = np.concatenate([verifier.score(pairs[start:start + 25])
                             for start in range(0, len(pairs), 25)])
    return values.reshape(len(rows), 5)


def decisions(rows, ranker, choices, values, threshold):
    counts = dict.fromkeys(("supported", "unsupported", "accepted_supported", "accepted_unsupported",
                           "supported_hit_at_1", "supported_hit_at_5", "candidate_recall_count"), 0)
    misses = []
    for row, indices, scores in zip(rows, choices, values, strict=True):
        names = [ranker.tools[index]["name"] for index in indices[np.argsort(-scores, kind="stable")]]
        accepted = bool(scores.max() >= threshold)
        if row["expected"]:
            counts["supported"] += 1
            present = row["expected"] in names
            counts["candidate_recall_count"] += present
            counts["accepted_supported"] += accepted
            counts["supported_hit_at_1"] += accepted and row["expected"] in names[:1]
            counts["supported_hit_at_5"] += accepted and present
            if not (accepted and present):
                misses.append(row["id"])
        else:
            counts["unsupported"] += 1
            counts["accepted_unsupported"] += accepted
    return {**counts, "missed_supported_ids": misses}


def cpu_profile(path, ranker, rows, choices):
    start = perf_counter()
    verifier = Verifier(path, "cpu")
    startup = perf_counter() - start
    timings = []
    # Measure one five-pair batch per request with no output generation.
    sample = [(row, ids) for row, ids in zip(rows, choices, strict=True)][::4]
    with torch.inference_mode():
        for i, (row, ids) in enumerate(sample):
            start = perf_counter()
            verifier.score([(row["query"], ranker.tools[index]["description"]) for index in ids])
            if i:
                timings.append(1000 * (perf_counter() - start))
    return {"threads": torch.get_num_threads(), "profiled_requests": len(timings),
            "model_load_ms": 1000 * startup, "warm_p50_ms": float(np.median(timings)),
            "warm_p95_ms": float(np.quantile(timings, .95))}


def run(spec, checkpoint, model, device, profile):
    base_protocol = json.loads((HERE / "protocol.json").read_text())
    protocol = json.loads((HERE / "capability_protocol.json").read_text())
    if fingerprint(spec) != base_protocol["spec_sha256"]:
        raise ValueError("OpenAPI snapshot hash differs from this pilot")
    tools = catalog_from_openapi(json.loads(spec.read_text()), "weather")
    names = {tool["name"] for tool in tools}
    calibration_file, holdout_file = HERE / "requests.tsv", HERE / "capability_holdout.tsv"
    dev = [row for row in read_requests(calibration_file, names) if row["split"] == "calibration"]
    test = read_requests(holdout_file, names)
    assert all(row["split"] == "holdout" for row in test)
    assert not {row["query"].casefold() for row in dev} & {row["query"].casefold() for row in test}
    ranker = OnnxToolRank(checkpoint, tools, hybrid=True)
    dev_rows, dev_choices = shortlist(ranker, dev)
    test_rows, test_choices = shortlist(ranker, test)
    verifier = Verifier(model, device)
    dev_margins = model_scores(verifier, ranker, dev, dev_choices).max(axis=1)
    scores_by_id = {row["id"]: margin for row, margin in zip(dev, dev_margins, strict=True)}
    cosine_by_id = {row["id"]: row["features"]["cosine"] for row, _ in dev_rows}
    verifier_cutoff = calibrate(labels(dev), max_false_accept=.1, confidence=.95,
                                score_requests=lambda ids: [scores_by_id[i] for i in ids])
    cosine_cutoff = calibrate(labels(dev), max_false_accept=.1, confidence=.95,
                              score_requests=lambda ids: [cosine_by_id[i] for i in ids])
    test_margins = model_scores(verifier, ranker, test, test_choices)
    evaluated = decisions(test, ranker, test_choices, test_margins, verifier_cutoff["min_score"])
    del verifier
    timing = cpu_profile(model, ranker, dev, dev_choices) if profile else None
    gate = protocol["promotion_gate"]
    passed = {
        "support_with_gold": evaluated["supported_hit_at_5"] / evaluated["supported"]
        >= gate["min_fraction_supported_with_gold_at_five"],
        "false_acceptance": evaluated["accepted_unsupported"] / evaluated["unsupported"]
        <= gate["max_fraction_unsupported_accepted"],
        "cpu_latency": timing["warm_p95_ms"] <= gate["max_cpu_p95_ms_for_span_of_five_tools"] if timing else None,
    }
    return {
        "model": "Qwen/Qwen3-Reranker-0.6B", "revision": REVISION,
        "spec_sha256": fingerprint(spec), "calibration_sha256": fingerprint(calibration_file),
        "holdout_sha256": fingerprint(holdout_file),
        "protocol_sha256": fingerprint(HERE / "capability_protocol.json"),
        "source_sha256": fingerprint(Path(__file__)),
        "scorer_sha256": fingerprint(HERE / "verifier.py"),
        "encoder_checksums": checkpoint_checksums(checkpoint, "embedding-int8.onnx"),
        "torch": torch.__version__, "device": device, "cpu_profile": timing,
        "calibration": {"verifier": verifier_cutoff, "cosine": cosine_cutoff},
        "holdout": {"verifier": evaluated, "cosine": evaluate(test_rows, ranker.tools, "hybrid",
                                                              "cosine", cosine_cutoff["min_score"]),
                    "unfiltered": evaluate(test_rows, ranker.tools, "hybrid")},
        "gate": {"passed": all(passed.values()), "checks": passed}, "external_compute_usd": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--profile-cpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.write_text(json.dumps(run(args.spec, args.checkpoint, args.model, args.device, args.profile_cpu),
                                     indent=2) + "\n")
