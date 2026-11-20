"""Replay catalog-specific rejection cutoffs with the released INT8 ranker."""
import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

from examples.catalog_lexical import top_indices, words
from examples.openapi_catalog import catalog_from_openapi
from examples.rank_onnx import OnnxToolRank, calibrate, checkpoint_checksums


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("abstention_replay_protocol.json")
NWS = ROOT / "tests/nws_pilot"
LIBRARY = ROOT / "data/onnx_calibration_demo.json"
API = ROOT / "data/onnx_int8_holdout.json"


@dataclass
class Catalog:
    tools: list
    calibration: list
    holdout: list


@dataclass
class QueryScores:
    expected: set
    cosine: np.ndarray
    hybrid: np.ndarray
    bm25: np.ndarray
    signals: dict


def catalogs():
    spec = gzip.decompress((NWS / "spec.json.gz").read_bytes())
    expected_hash = json.loads((NWS / "protocol.json").read_text())["spec_sha256"]
    if sha256(spec).hexdigest() != expected_hash:
        raise ValueError("NWS snapshot hash differs from the published protocol")
    tools = catalog_from_openapi(json.loads(spec), "weather")
    with (NWS / "requests.tsv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))

    def nws_queries(split):
        return [(row["query"], set() if row["operation"] == "-" else {"weather__" + row["operation"]})
                for row in rows if row["split"] == split]

    library = json.loads(LIBRARY.read_text())

    def library_queries(split):
        return [(row["query"], set(row["relevant"])) for row in library[split]]

    api = json.loads(API.read_text())
    api_queries = [(row["query"], {api["tools"][key]["name"] for key in row["positives"]})
                   for row in api["queries"]]
    return {"nws": Catalog(tools, nws_queries("calibration"), nws_queries("holdout")),
            "library": Catalog(library["catalog"]["tools"], library_queries("calibration"),
                               library_queries("evaluation")),
            "api": Catalog(list(api["tools"].values()), [], api_queries)}


class QueryCoverage:
    """IDF-weighted fraction of distinct query terms supported by a document."""

    def __init__(self, documents):
        self.documents = [set(words(document)) for document in documents]
        frequency = Counter(term for document in self.documents for term in document)
        unseen = 1 + math.log(len(documents) + 1)
        self.weights = {term: unseen - math.log(count + 1) for term, count in frequency.items()}
        self.unseen = unseen

    def scores(self, query, indices):
        tokens = {word: self.weights.get(word, self.unseen) for word in sorted(set(words(query)))}
        total = sum(tokens.values())
        return np.array([sum(weight for word, weight in tokens.items() if word in self.documents[index]) / total
                         if total else 0 for index in indices], dtype=np.float32)


def score_queries(ranker, coverage, cases, protocol):
    ranking = protocol["ranking"]
    k = ranking["top_k"]
    samples = []
    for query, expected in cases:
        cosine = ranker.cosine_scores(query)
        tfidf, bm25 = ranker.lexical_index.scores(query)
        hybrid = cosine + ranking["tfidf"] * tfidf + ranking["bm25"] * bm25
        best = top_indices(hybrid, k)
        lexical = top_indices(bm25, k)
        lexical = lexical[bm25[lexical] > 0]
        support = coverage.scores(query, best)
        lexical_support = coverage.scores(query, lexical)
        second, first = np.partition(cosine, -2)[-2:]
        signals = {
            "cosine": first,
            "hybrid_max": hybrid[best[0]],
            "cosine_margin": first - second,
            "cosine_z": (first - cosine.mean()) / max(cosine.std(), 1e-6),
            "hybrid_query_coverage": support.max(initial=0),
            "bm25_query_coverage": lexical_support.max(initial=0),
            "cosine_plus_coverage": (cosine[best] + protocol["calibration"]["coverage_boost"] * support).max(),
        }
        samples.append(QueryScores(expected, cosine, hybrid, bm25, signals))
    return samples


def scope_metrics(samples, names, method, k, threshold=None):
    supported = unsupported = hits_1 = hits_k = accepted_supported = accepted_unsupported = 0
    all_indices = np.arange(len(names))
    for row in samples:
        indices = all_indices
        ranking = row.hybrid
        if method.startswith("bm25"):
            ranking = row.bm25
            indices = indices[row.bm25 > 0]
        if threshold is not None:
            if method == "cosine":
                # Exact deployed filter. Each returned candidate must pass the cutoff.
                indices = indices[row.cosine[indices] >= threshold]
            elif np.float32(row.signals[method]) < threshold:
                indices = indices[:0]
        picked = names[indices[top_indices(ranking[indices], k)]]
        if row.expected:
            supported += 1
            accepted_supported += bool(len(picked))
            hits_1 += bool(row.expected.intersection(picked[:1]))
            hits_k += bool(row.expected.intersection(picked))
        else:
            unsupported += 1
            accepted_unsupported += bool(len(picked))
    accepted = accepted_supported + accepted_unsupported
    return {"supported": supported, "unsupported": unsupported, "accepted_supported": accepted_supported,
            "accepted_unsupported": accepted_unsupported, "supported_hits_at_1": hits_1,
            "supported_hits_at_5": hits_k, "supported_hit_at_5": hits_k / supported,
            "supported_coverage": accepted_supported / supported, "false_acceptance": accepted_unsupported / unsupported,
            "selective_hit_at_5": hits_k / accepted if accepted else None}


def validate_catalog(name, catalog, names):
    names = set(names)
    development = {query.strip().casefold() for query, _ in catalog.calibration}
    holdout = {query.strip().casefold() for query, _ in catalog.holdout}
    if development & holdout:
        raise ValueError(f"Repeated calibration/holdout request in {name}")
    if any(gold - names for _, gold in catalog.calibration + catalog.holdout):
        raise ValueError(f"Invalid relevant tool for {name}")


def replay(checkpoint, protocol):
    inputs = [NWS / "spec.json.gz", NWS / "requests.tsv", NWS / "protocol.json", LIBRARY, API, PROTOCOL]
    records = {}
    start = perf_counter()
    ranking = protocol["ranking"]
    k = ranking["top_k"]
    calibrated = {}
    catalog_data = catalogs()
    fingerprints = checkpoint_checksums(checkpoint, "embedding-int8.onnx")
    if fingerprints["onnx/embedding-int8.onnx"] != protocol["checkpoint_sha256"]:
        raise ValueError("Use the INT8 graph pinned in the protocol")

    # Complete all fits before scoring any holdout.
    for name, catalog in catalog_data.items():
        started = perf_counter()
        ranker = OnnxToolRank(checkpoint, catalog.tools, hybrid=True, threads=ranking["threads"])
        names = np.array([tool["name"] for tool in ranker.tools])
        validate_catalog(name, catalog, names)
        support = QueryCoverage([f"{tool['name']}: {tool.get('description') or ''}" for tool in ranker.tools])
        records[name] = {"tools": len(names), "startup_and_index_ms": (perf_counter() - started) * 1000}
        catalog_data[name] = (catalog, ranker, names, support)
        if not catalog.calibration:
            continue
        scored = score_queries(ranker, support, catalog.calibration, protocol)
        calibrated[name] = {}
        cases = [(str(index), row.expected) for index, row in enumerate(scored)]
        calibration = protocol["calibration"]
        for signal in protocol["signals"][2:]:
            fit = calibrate(cases, score_requests=lambda idx: [scored[int(i)].signals[signal] for i in idx],
                            max_false_accept=calibration["max_false_accept"], confidence=calibration["confidence"])
            negative = np.array([row.signals[signal] for row in scored if not row.expected], dtype=np.float32)
            if fit["min_score"] is None:
                raise ValueError(f"Insufficient calibration labels in {name}")
            points = np.quantile(negative, calibration["negative_quantiles_for_diagnostic_curve"]).astype(np.float32)
            calibrated[name][signal] = (fit, sorted({float(value) for value in points}))

    for name, (catalog, ranker, names, support) in catalog_data.items():
        started = perf_counter()
        samples = score_queries(ranker, support, catalog.holdout, protocol)
        record = records[name]
        record["holdout_scoring_seconds"] = perf_counter() - started
        record["unfiltered"] = {method: scope_metrics(samples, names, method, k)
                                for method in protocol["signals"][:2]}
        record["cutoffs_fitted_from"] = {}
        for origin, methods in calibrated.items():
            record["cutoffs_fitted_from"][origin] = {
                method: {"fit": fit, "holdout": scope_metrics(samples, names, method, k, fit["min_score"]),
                         "calibration_quantile_curve": [
                             {"threshold": point, **scope_metrics(samples, names, method, k, point)} for point in points
                         ]}
                for method, (fit, points) in methods.items()
            }

    gate = protocol["gate"]
    local_passes = {
        method: {name: (record["false_acceptance"] <= gate["max_false_accept"]
                        and record["supported_coverage"] >= gate["min_supported_coverage"]
                        and record["supported_hit_at_5"] >= gate["min_supported_hit_at_5"])
                 for name, result in records.items() if name in calibrated
                 for record in [result["cutoffs_fitted_from"][name][method]["holdout"]]}
        for method in protocol["signals"][2:]
    }
    return {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "platform": platform.platform(), "numpy": np.__version__,
            "checkpoint_sha256": fingerprints, "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256": {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest() for path in inputs},
            "protocol": protocol, "catalogs": records, "passes_local_gate": local_passes,
            "passing_all_local_catalogs": [method for method, passes in local_passes.items() if all(passes.values())],
            "wall_seconds": perf_counter() - start, "external_compute_usd": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    results = replay(options.checkpoint, json.loads(PROTOCOL.read_text()))
    options.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
