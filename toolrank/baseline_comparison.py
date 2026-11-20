"""Reproduce the matched dense and lexical tool retrieval comparisons."""
import argparse
from collections import defaultdict
from hashlib import file_digest
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

from examples.catalog_lexical import top_indices
from toolrank.benchmark import encode
from toolrank.core import BGE, REX, SCOUT, load_model, metrics, write_json
from toolrank.etom import FILES, REVISION
from toolrank.fetch import REVISIONS
from toolrank.lexical import Bm25Index, lexical_text
from toolrank.replay_hybrid import HOLDOUT, corpus


DATASETS = ("etom-plain", "etom-mcp", "etom-openai", "toolret", "api")
ETOM_ROOT = Path(".work/next/etom")
MODELS = {"bge": BGE, "toolrank": "christianengel/mcp-toolrank-33m", "toolscout": SCOUT,
          "tool-embed": REX}
TOOLRANK_REVISION = "08b109f86e88c43304732fcb45432dc67c294cdb"
WEIGHTS = {"tfidf": 0.4, "bm25": 0.15}
SEED = 20261120


def digest(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


class Lexical:
    def __init__(self, texts):
        self.bm25 = Bm25Index(texts)
        self.tfidf = TfidfVectorizer(preprocessor=lexical_text, ngram_range=(1, 2),
                                     dtype=np.float32)
        self.documents = self.tfidf.fit_transform(texts)

    def score(self, queries):
        tfidf = (self.tfidf.transform(queries) @ self.documents.T).toarray()
        return tfidf, self.bm25.scores(queries)


def rankings(dense, tfidf, bm25, hybrids=("bge", "toolrank")):
    yield "bm25", bm25
    yield "tfidf", tfidf
    for name, similarity in dense.items():
        yield f"{name}-dense", similarity
        if name in hybrids:
            hybrid = similarity + WEIGHTS["tfidf"] * tfidf
            yield f"{name}-tfidf", hybrid
            yield f"{name}-hybrid", hybrid + WEIGHTS["bm25"] * bm25


def measure(order, gold):
    result = metrics(order, gold)
    for k in (1, 5, 10):
        result[f"hit@{k}"] = float(any(key in gold for key in order[:k]))
        result[f"complete@{k}"] = float(gold.keys() <= set(order[:k]))
    return result


def paired_interval(candidate, baseline, groups, seed=SEED, samples=3000):
    """Resample whole groups and retain pairing and query-weighted aggregation."""
    differences = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    _, clusters = np.unique(groups, return_inverse=True)
    counts = np.bincount(clusters)
    totals = np.bincount(clusters, weights=differences)
    selected = np.random.default_rng(seed).integers(len(counts), size=(samples, len(counts)))
    bootstrap = totals[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return {"delta": float(differences.mean()),
            "ci95": np.quantile(bootstrap, [.025, .975]).tolist(), "clusters": len(counts)}


def fingerprint(dataset, etom_root=ETOM_ROOT):
    if dataset == "toolret":
        files = sorted(Path(".work/toolret").glob("*/*.parquet"))
        versions = {key: value for key, value in REVISIONS.items() if key.startswith("datasets/")}
    elif dataset.startswith("etom-"):
        files = [etom_root / name for name in FILES]
        versions = {"etom": REVISION}
    else:
        files = [HOLDOUT]
        versions = {"api": "repository fixture"}
    return {"versions": versions, "sha256": {str(path): digest(path) for path in files}}


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def compare(dataset, models, device, batch_size, paired_models=("bge", "toolrank")):
    started = perf_counter()
    docs, queries = corpus(dataset, ETOM_ROOT)
    keys = np.asarray(sorted(docs))
    texts = [docs[key] for key in keys]
    judged = []
    groups = []
    missing = 0
    for _, text, family, gold in queries:
        positive = {key: value for key, value in gold.items() if value > 0}
        if positive.keys() - docs.keys():
            missing += 1
            continue
        if positive:
            if dataset.startswith("etom-"):
                family = min(key.split("::", 1)[0] for key in positive)
            judged.append((text, positive))
            groups.append(family)
    questions, labels = zip(*judged, strict=True)
    lex_start = perf_counter()
    lexical = Lexical(texts)
    lexical_seconds = perf_counter() - lex_start
    vectors = {}
    indexing = {}
    for name, model in models.items():
        begin = perf_counter()
        documents = encode(model, texts, batch_size)
        synchronize(device)
        indexing[name] = {"seconds": perf_counter() - begin}
        vectors[name] = (documents, encode(model, list(questions), batch_size, query=True))
    per_row = defaultdict(list)
    for start in range(0, len(judged), 64):
        subset = slice(start, start + 64)
        tfidf, bm25 = lexical.score(questions[subset])
        dense = {name: (query[subset] @ document.T).cpu().numpy()
                 for name, (document, query) in vectors.items()}
        for name, scores in rankings(dense, tfidf, bm25, paired_models):
            # Stable ties use the same sorted tool IDs across systems.
            per_row[name].extend(measure(keys[top_indices(row, 10)], gold)
                                 for row, gold in zip(scores, labels[subset], strict=True))
    results = {name: {metric: float(np.mean([row[metric] for row in rows]))
                      for metric in rows[0]} for name, rows in per_row.items()}
    paired = {}
    baseline, candidate = paired_models
    if {baseline, candidate} <= models.keys():
        for method in ("dense", "tfidf", "hybrid"):
            paired[method] = {
                metric: paired_interval([row[metric] for row in per_row[f"{candidate}-{method}"]],
                                        [row[metric] for row in per_row[f"{baseline}-{method}"]], groups)
                for metric in ("ndcg@10", "hit@1", "recall@5", "complete@5")
            }
    return {"tools": len(docs), "queries": len(queries), "answerable_scored": len(judged),
            "unavailable_judgments": missing, "no_match": len(queries) - len(judged) - missing,
            "metrics": results, f"{candidate}_minus_{baseline}": paired, "embedding_index": indexing,
            "lexical_index_seconds": lexical_seconds, "wall_seconds": perf_counter() - started,
            "dataset": fingerprint(dataset)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolrank", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--models", choices=tuple(MODELS), nargs="+", default=list(MODELS))
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=list(DATASETS))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    models = {}
    metadata = {}
    for name in args.models:
        model_id = MODELS[name]
        target = str(args.toolrank.resolve()) if name == "toolrank" else model_id
        started = perf_counter()
        model = load_model(target, args.device, {"dtype": torch.float32})
        model.max_seq_length = 512
        synchronize(args.device)
        models[name] = model
        weights = (args.toolrank if name == "toolrank" else Path(".work/models") / target) / "model.safetensors"
        metadata[name] = {"model": model_id, "revision": TOOLRANK_REVISION if name == "toolrank"
                          else REVISIONS[f"models/{model_id}"],
                          "weights_sha256": digest(weights), "load_seconds": perf_counter() - started,
                          "parameters": sum(parameter.numel() for parameter in model.parameters()),
                          "query_prompt": model.prompts.get("query", "")}
    output = {"models": metadata, "weights": WEIGHTS, "precision": "float32", "max_sequence_length": 512,
              "weight_selection": "Previously selected on provider-disjoint development only",
              "paired_bootstrap_seed": SEED, "paired_bootstrap_samples": 3000, "device": args.device,
              "platform": platform.platform(), "threads": 4, "batch_size": args.batch,
              "torch_version": torch.__version__,
              "code_base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "benchmark_source_sha256": digest(Path(__file__)),
              "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None, "datasets": {}}
    for dataset in args.datasets:
        result = compare(dataset, models, args.device, args.batch)
        output["datasets"][dataset] = result
        write_json(args.output, output)
        print(dataset, {name: round(values["ndcg@10"] * 100, 3)
                        for name, values in result["metrics"].items()}, flush=True)


if __name__ == "__main__":
    main()
