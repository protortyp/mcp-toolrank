"""Compare float32 and compact CPU scoring over frozen ToolRet and ETOM corpora."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path
import platform
from statistics import mean
import time

from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import onnxruntime as ort
from threadpoolctl import threadpool_limits
import torch

from toolrank.benchmark import toolret
from toolrank.core import load_model, metrics, write_json
from toolrank import etom
from toolrank.fetch import REVISIONS, download

from examples.rank_onnx import CompactCatalog, top_indices


def fetch_evaluation():
    for repo, revision in REVISIONS.items():
        if not repo.startswith("datasets/"):
            continue
        repo_id = repo.removeprefix("datasets/")
        files = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision, token=False)
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda file: hf_hub_download(
                repo_id, file, repo_type="dataset", revision=revision, local_dir=".work/toolret", token=False,
            ), (file for file in files if file.endswith(".parquet"))))
    upstream = f"https://raw.githubusercontent.com/snooow1029/ETOM/{etom.REVISION}/data"
    for name in etom.FILES:
        prefix = "queries/" if name.startswith("level_") else ""
        download((etom.ROOT / name, f"{upstream}/{prefix}{name}"))


def evaluate(model, docs, rows, threads, cutoff=None):
    ids = sorted(docs)
    opts = dict(batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    document_vectors = model.encode_document([docs[key] for key in ids], **opts)
    query_vectors = model.encode_query([row[1] for row in rows], **opts)
    index = CompactCatalog((document_vectors[i:i+32] for i in range(0, len(ids), 32)), len(ids), threads)
    methods = {"float32": lambda q: q @ document_vectors.T, "int8": index.scores}
    by_method = {}
    ranked = {}
    for name, score in methods.items():
        stats = defaultdict(list)
        ranked[name] = []
        for i in range(0, len(query_vectors), 64):
            scores = score(query_vectors[i:i+64])
            for row, values in zip(rows[i:i+64], scores, strict=True):
                top = top_indices(values, 10)
                keys = [ids[index] for index in top]
                ranked[name].append(keys)
                gold = row[3]
                relevant = any(value > 0 for value in gold.values())
                if relevant:
                    values_by_metric = metrics(keys, gold)
                    for key, value in values_by_metric.items():
                        stats[key].append(value)
                if cutoff is not None:
                    status = "answerable_acceptance" if relevant else "no_match_false_acceptance"
                    stats[status].append(float(values[top[0]] >= cutoff))
        timings = []
        for query in query_vectors[:16]:
            top_indices(score(query[None, :])[0], 10)
        for query in query_vectors[:384]:
            start = time.perf_counter()
            top_indices(score(query[None, :])[0], 10)
            timings.append((time.perf_counter() - start) * 1000)
        by_method[name] = {
            "metrics": {key: mean(values) for key, values in stats.items()},
            "score_and_top10_p50_ms": float(np.median(timings)),
            "score_and_top10_p95_ms": float(np.quantile(timings, 0.95)),
            "catalog_storage_bytes": (document_vectors.nbytes if name == "float32"
                                      else index.vectors.nbytes + index.scales.nbytes),
        }
    return {
        "documents": len(docs), "queries": len(rows), "cutoff": cutoff,
        "top1_agreement": mean(a[0] == b[0] for a, b in zip(*ranked.values(), strict=True)),
        "mean_top10_overlap": mean(len(set(a) & set(b)) / 10 for a, b in zip(*ranked.values(), strict=True)),
        "results": by_method,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="path to the pinned SentenceTransformer checkpoint")
    parser.add_argument("--device", default="cuda", help="device that embeds queries and descriptions")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch-evaluation", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.fetch_evaluation:
        fetch_evaluation()
    torch.set_num_threads(args.threads)
    model = load_model(args.checkpoint, args.device)
    with threadpool_limits(limits=args.threads):
        docs, rows = toolret()
        toolret_results = evaluate(model, docs, rows, args.threads)
        docs, rows, _ = etom.load("plain")
        etom_results = evaluate(model, docs, rows, args.threads, cutoff=0.503631055355072)
    write_json(args.output, {
        "encoder_device": (torch.cuda.get_device_name() if args.device == "cuda" else "cpu"),
        "onnxruntime": ort.__version__,
        "platform": platform.platform(),
        "threads": args.threads,
        "toolret": toolret_results,
        "etom": etom_results,
    })


if __name__ == "__main__":
    main()
