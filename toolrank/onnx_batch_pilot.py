"""Compare batch sensitivity, ranking, and CPU cost of pinned ONNX variants."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import platform
from subprocess import check_output
import sys
from time import perf_counter

from toolrank.core import metrics, read_json, write_json
from toolrank.etom import ROOT as ETOM_ROOT, load

PROTOCOL = Path(__file__).with_name("onnx_batch_protocol.json")
DEV = Path("data/dev.json")
API = Path("data/onnx_int8_holdout.json")
FILES = {"fp32": "embedding.onnx", "dynamic-int8": "embedding-int8.onnx",
         "static-int8": "embedding-static-int8.onnx", "weight-int4": "embedding-weight-int4.onnx"}


def digest(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def fixture(dataset, root):
    if dataset == "etom-plain":
        docs, cases, stamp = load("plain", root)
        cases = [(text, min(key.partition("::")[0] for key in gold), gold)
                 for _, text, _, gold in cases if gold]
        return docs, cases, stamp
    payload = read_json(API)
    docs = {key: f"{row['name']}: {row['description']}" for key, row in payload["tools"].items()}
    cases = [(row["query"], row["family"], dict.fromkeys(row["positives"], 1))
             for row in payload["queries"] if row["positives"]]
    return docs, cases, {"sha256": digest(API), "unanswerable": len(payload["queries"]) - len(cases)}


def dev_probe(encoder):
    import numpy as np

    groups = read_json(DEV)
    tools = [row for group in groups for row in group["tools"]]
    queries = [encoder.prefix + text for row in tools for text in row["queries"]]
    docs = [f"{row['name']}: {row['description']}" for row in tools]
    partner = (" ".join(docs[-3:]) + " ") * 20
    deltas = []
    for text in docs[:4] + queries[:4]:
        alone = encoder.encode([text])[0]
        mixed = encoder.encode([docs[-1], text, partner])[1]
        deltas.append(float(np.max(np.abs(alone - mixed))))
    times = []
    for _ in range(4):
        encoder.encode([queries[0]])
    for i in range(64):
        start = perf_counter()
        encoder.encode([queries[i % len(queries)]])
        times.append(1000 * (perf_counter() - start))
    return {"sha256": digest(DEV), "mixed_pair_count": len(deltas), "max_embedding_delta": max(deltas),
            "p50_query_ms": float(np.median(times)), "p95_query_ms": float(np.quantile(times, .95)),
            "long_partner_tokens": len(encoder.tokenizer.encode(partner).ids)}


def score_dataset(encoder, docs, cases, batch):
    import numpy as np

    from examples.catalog_lexical import CatalogLexicalIndex, top_indices

    keys = np.asarray(sorted(docs))
    texts = [docs[key] for key in keys]
    lexical = CatalogLexicalIndex(texts)
    orders = {"stable": list(range(len(texts))), "bucketed": sorted(range(len(texts)), key=lambda i: len(texts[i]))}
    matrices = {}
    timings = {}
    for name, order in orders.items():
        start = perf_counter()
        rows = np.empty((len(texts), 384), dtype=np.float32)
        for i in range(0, len(order), batch):
            chunk = order[i:i + batch]
            rows[chunk] = encoder.encode([texts[index] for index in chunk])
        matrices[name] = rows
        timings[name] = perf_counter() - start
    measures = {order: {"dense": [], "hybrid": []} for order in orders}
    equal = dict.fromkeys(("dense", "hybrid"), 0)
    for query, _, gold in cases:
        vector = encoder.encode([encoder.prefix + query])[0]
        tfidf, bm25 = lexical.scores(query)
        ranked = {}
        for order, matrix in matrices.items():
            cosine = np.einsum("ij,j->i", matrix, vector)
            for ranker, scores in (("dense", cosine), ("hybrid", cosine + .4 * tfidf + .15 * bm25)):
                ids = keys[top_indices(scores, 10)].tolist()
                measures[order][ranker].append(metrics(ids, gold))
                if order == "stable":
                    ranked[ranker] = ids
                else:
                    equal[ranker] += ids == ranked[ranker]
    return {"tools": len(docs), "supported": len(cases), "index_seconds": timings,
            "max_catalog_embedding_delta": float(np.max(np.abs(matrices["stable"] - matrices["bucketed"]))),
            "same_top10_fraction": {key: value / len(cases) for key, value in equal.items()},
            "per_query": measures}


def worker(args, protocol):
    import onnxruntime

    from examples.rank_onnx import OnnxToolRank, checkpoint_checksums

    name = args.worker
    root = {"fp32": args.fp32, "dynamic-int8": args.int8}.get(name, args.candidates)
    before = perf_counter()
    encoder = OnnxToolRank(root, [], threads=protocol["threads"], model_file=FILES[name])
    graph = root / "onnx" / FILES[name]
    metadata = {"load_seconds": perf_counter() - before, "graph_bytes": graph.stat().st_size,
                "checkpoint": checkpoint_checksums(root, FILES[name]), "ort": onnxruntime.__version__,
                "development": dev_probe(encoder), "datasets": {}}
    for dataset in args.datasets:
        docs, cases, _ = fixture(dataset, args.etom)
        metadata["datasets"][dataset] = score_dataset(encoder, docs, cases, protocol["batch"])
    rss = next(int(line.split()[1]) for line in Path("/proc/self/status").read_text().splitlines()
               if line.startswith("VmHWM:"))
    return metadata | {"peak_rss_mib": rss / 1024}


def compare(args, protocol):
    results = {}
    for name in FILES:
        command = [sys.executable, "-m", "toolrank.onnx_batch_pilot", "--worker", name, "--fp32", str(args.fp32),
                   "--int8", str(args.int8), "--candidates", str(args.candidates), "--etom", str(args.etom), "--datasets",
                   *args.datasets]
        results[name] = json.loads(check_output(command, timeout=420, text=True))
        print(name, results[name]["development"], flush=True)
    from toolrank.baseline_comparison import paired_interval

    data = {}
    for dataset in args.datasets:
        _, cases, stamp = fixture(dataset, args.etom)
        data[dataset] = stamp
        groups = [group for _, group, _ in cases]
        reference = results["dynamic-int8"]["datasets"][dataset]["per_query"]["stable"]
        for row in results.values():
            record = row["datasets"][dataset]
            per_query = record.pop("per_query")
            record["metrics"] = {}
            for order, rankings in per_query.items():
                record["metrics"][order] = {}
                for ranker, rows in rankings.items():
                    relative = paired_interval([sample["ndcg@10"] for sample in rows],
                                                [sample["ndcg@10"] for sample in reference[ranker]], groups,
                                                seed=protocol["bootstrap_seed"])
                    means = {key: sum(sample[key] for sample in rows) / len(rows) for key in rows[0]}
                    record["metrics"][order][ranker] = means | {"ndcg_delta_vs_dynamic_stable": relative}
    output = {"revision": check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": digest(__file__), "protocol_sha256": digest(PROTOCOL),
              "platform": platform.platform(), "cpu": next(line.split(":", 1)[1].strip()
                  for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
              "export": {name: read_json(args.candidates / f"{name}.json")
                         for name in protocol["candidates"]}, "dataset_stamps": data,
              "results": results, "external_compute_usd": 0}
    write_json(args.output, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fp32", "int8", "candidates"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--etom", type=Path, default=ETOM_ROOT)
    parser.add_argument("--datasets", nargs="*", choices=("api", "etom-plain"), default=("api", "etom-plain"))
    parser.add_argument("--worker", choices=FILES, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = read_json(PROTOCOL)
    if args.worker:
        print(json.dumps(worker(args, protocol)))
    elif args.output:
        compare(args, protocol)
    else:
        parser.error("--output is required")


if __name__ == "__main__":
    main()
