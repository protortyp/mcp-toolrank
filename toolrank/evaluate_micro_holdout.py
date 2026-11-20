"""Evaluate the one development-selected INT8 micro checkpoint against ToolRank."""
import argparse
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from tempfile import TemporaryDirectory

import numpy as np
from threadpoolctl import threadpool_limits

from examples.catalog_lexical import CatalogLexicalIndex, top_indices
from examples.rank_onnx import (OnnxToolRank, catalog_cache_path, checkpoint_checksums,
                                read_cached_vectors, write_cached_vectors)
from toolrank.baseline_comparison import fingerprint, measure, paired_interval
from toolrank.core import read_json, write_json
from toolrank.profile_micro_distillation import GRAPH, variant
from toolrank.replay_hybrid import corpus
from toolrank.train_micro_distillation import digest

PROTOCOL = Path(__file__).with_name("micro_holdout_protocol.json")


def catalog_vectors(runtime, texts, batch, cache_path=None):
    if cache_path is not None:
        cached = read_cached_vectors(cache_path, len(texts), 384, "float32")
        if cached is not None:
            return cached[0], True
    vectors = np.concatenate([runtime.encode(texts[i:i + batch]) for i in range(0, len(texts), batch)])
    if cache_path is not None:
        write_cached_vectors(cache_path, (vectors, None))
    return vectors, False


def compare(catalog, cases, runtimes, batch, cache_paths=None):
    keys = np.asarray(sorted(catalog))
    texts = [catalog[key] for key in keys]
    judged = []
    unavailable = unsupported = 0
    for _, text, group, gold in cases:
        positives = {key: value for key, value in gold.items() if value > 0}
        if positives.keys() - catalog.keys():
            unavailable += 1
        elif not positives:
            unsupported += 1
        else:
            judged.append((text, group, positives))
    lexical = CatalogLexicalIndex(texts)
    matrices, indexing, cache_hits = {}, {}, {}
    cache_paths = cache_paths or {}
    for name, runtime in runtimes.items():
        start = perf_counter()
        matrices[name], cache_hits[name] = catalog_vectors(runtime, texts, batch, cache_paths.get(name))
        indexing[name] = perf_counter() - start
    metrics = {name: {"dense": [], "hybrid": []} for name in runtimes}
    with threadpool_limits(limits=4, user_api="blas"):
        for start in range(0, len(judged), 64):
            block = judged[start:start + 64]
            lexical_weights = []
            for query, _, _ in block:
                tfidf, bm25 = lexical.scores(query)
                lexical_weights.append(.4 * tfidf + .15 * bm25)
            blend = np.asarray(lexical_weights)
            for name, runtime in runtimes.items():
                vectors = np.concatenate([runtime.encode([runtime.prefix + query]) for query, _, _ in block])
                cosine = vectors @ matrices[name].T
                for ranker, scores in (("dense", cosine), ("hybrid", cosine + blend)):
                    metrics[name][ranker].extend(measure(keys[top_indices(row, 10)].tolist(), gold)
                                                 for row, (_, _, gold) in zip(scores, block, strict=True))
    means = {name: {ranker: {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
                    for ranker, rows in variants.items()} for name, variants in metrics.items()}
    groups = [group for _, group, _ in judged]
    paired = {}
    for ranker in ("dense", "hybrid"):
        paired[ranker] = {}
        for metric in ("ndcg@10", "hit@5", "complete@10"):
            paired[ranker][metric] = paired_interval([row[metric] for row in metrics["micro"][ranker]],
                                                       [row[metric] for row in metrics["toolrank"][ranker]], groups)
    return {"tools": len(keys), "answerable": len(judged), "unsupported": unsupported,
            "unavailable_judgments": unavailable, "index_seconds": indexing, "catalog_cache_hits": cache_hits,
            "metrics": means,
            "micro_minus_toolrank": paired}


def run(args):
    protocol = read_json(PROTOCOL)
    profile = read_json(args.cpu_profile)
    if profile["student"] != protocol["candidate"]:
        raise ValueError("The CPU profile must use the selected candidate")
    main_start = perf_counter()
    output = {"protocol": protocol, "protocol_sha256": digest(PROTOCOL), "platform": platform.platform(),
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": digest(__file__), "cpu_profile_sha256": digest(args.cpu_profile),
              "datasets": {}, "external_compute_usd": 0}
    with TemporaryDirectory() as temporary:
        root = Path(temporary) / "micro"
        variant(args.models / protocol["candidate"], root, Path(temporary) / "export.json")
        paths = {"micro": root, "toolrank": args.int8}
        for name, key in (("micro", "candidate_int8_sha256"), ("toolrank", "baseline_int8_sha256")):
            if digest(paths[name] / "onnx" / GRAPH) != protocol[key]:
                raise ValueError(f"Unexpected {name} INT8 graph")
        output["checkpoints"] = {name: checkpoint_checksums(path, GRAPH) for name, path in paths.items()}
        runtimes = {name: OnnxToolRank(path, [], threads=protocol["threads"]) for name, path in paths.items()}
        for name, gate in protocol["datasets"].items():
            started = perf_counter()
            catalog, cases = corpus(name, args.etom)
            if name.startswith("etom-"):
                cases = [(key, text, min((tool.partition("::")[0] for tool in labels), default=group), labels)
                         for key, text, group, labels in cases]
            batch = protocol["doc_batch"]
            cache = {}
            if args.catalog_cache is not None:
                # Dynamic INT8 depends on the other documents in each batch.
                directory = args.catalog_cache / f"doc-batch-{batch}"
                texts = [catalog[key] for key in sorted(catalog)]
                cache = {label: catalog_cache_path(directory, path, texts, GRAPH, "float32", protocol["threads"])
                         for label, path in paths.items()}
            comparators = compare(catalog, cases, runtimes, batch, cache)
            paired = comparators["micro_minus_toolrank"]["hybrid"]
            passed = paired["ndcg@10"]["ci95"][0] >= gate["ndcg_lower_bound"] and (
                paired["hit@5"]["delta"] >= -protocol["max_hit5_point_loss"])
            output["datasets"][name] = comparators | {"fingerprint": fingerprint(name, args.etom), "passes": passed,
                                                        "wall_seconds": perf_counter() - started}
            write_json(args.output, output)
            print(name, "hybrid nDCG delta", paired["ndcg@10"], "gate", passed, flush=True)
    speed = {workload: row[protocol["candidate"]]["p95_ms"] / row["toolrank"]["p95_ms"]
             for workload, row in profile["profile"].items()}
    output["p95_ratios"] = speed
    size_mib = profile["graphs"][protocol["candidate"]]["bytes"] / 1024**2
    output["graph_mib"] = size_mib
    output["passes"] = (all(row["passes"] for row in output["datasets"].values())
                        and max(speed.values()) <= protocol["max_p95_ratio"]
                        and size_mib <= protocol["max_graph_mib"])
    output["wall_seconds"] = perf_counter() - main_start
    write_json(args.output, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("int8", "models", "cpu-profile", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--etom", type=Path, default=Path(".work/next/etom"))
    parser.add_argument("--catalog-cache", type=Path,
                        help="reuse document embeddings with identical graphs, text order and batching")
    run(parser.parse_args())
