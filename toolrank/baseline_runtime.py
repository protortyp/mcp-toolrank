"""Profile matched retrieval in a fresh CPU process per model."""
import argparse
import os
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import torch

from toolrank.baseline_comparison import DATASETS, ETOM_ROOT, MODELS, Lexical, WEIGHTS
from toolrank.core import load_model, write_json
from toolrank.replay_hybrid import corpus


def measure(search, queries):
    first = perf_counter()
    search(queries[:1])
    first_ms = 1000 * (perf_counter() - first)
    for text in queries[:4]:
        search([text])
    warm = []
    for text in queries:
        begin = perf_counter()
        search([text])
        warm.append(1000 * (perf_counter() - begin))
    start = perf_counter()
    for index in range(0, len(queries), 16):
        search(queries[index:index + 16])
    return {"first_query_ms": first_ms, "warm_p50_ms": float(np.median(warm)),
            "warm_p95_ms": float(np.quantile(warm, .95)),
            "batch16_queries_per_second": len(queries) / (perf_counter() - start)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("lexical", *MODELS))
    parser.add_argument("--toolrank", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=("api", "etom-mcp", "toolret"))
    parser.add_argument("--queries", type=int, default=48)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = None
    start = perf_counter()
    if args.model != "lexical":
        path = str(args.toolrank.resolve()) if args.model == "toolrank" else MODELS[args.model]
        model = load_model(path, "cpu", {"dtype": torch.float32})
        model.max_seq_length = 512

    def embed(texts, is_query):
        encode = model.encode_query if is_query else model.encode_document
        return encode(texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=32,
                      show_progress_bar=False)

    output = {"model": args.model, "checkpoint_load_ms": 1000 * (perf_counter() - start),
              "threads": 4, "torch_version": torch.__version__, "cpu": next(
                  (row.split(":", 1)[1].strip() for row in Path("/proc/cpuinfo").read_text().splitlines()
                   if row.startswith("model name")), None),
              "cpu_affinity": len(os.sched_getaffinity(0)), "datasets": {}}
    for dataset in args.datasets:
        documents, records = corpus(dataset, ETOM_ROOT)
        texts = [documents[key] for key in sorted(documents)]
        indices = np.linspace(0, len(records) - 1, min(args.queries, len(records)), dtype=int)
        queries = [records[i][1] for i in indices]
        start = perf_counter()
        lexical = Lexical(texts)
        lexical_ms = 1000 * (perf_counter() - start)
        start = perf_counter()
        vectors = embed(texts, False) if model is not None else None
        modes = ("bm25", "tfidf") if model is None else ("dense", "hybrid")

        def search(batch, mode):
            if vectors is None:
                scores = lexical.bm25.scores(batch) if mode == "bm25" else (
                    lexical.tfidf.transform(batch) @ lexical.documents.T).toarray()
            else:
                scores = (embed(batch, True) @ vectors.T).numpy()
                if mode == "hybrid":
                    tfidf, bm25 = lexical.score(batch)
                    scores = scores + WEIGHTS["tfidf"] * tfidf + WEIGHTS["bm25"] * bm25
            return np.argsort(-scores, axis=1, kind="stable")[:, :10]

        entry = {"tools": len(texts), "profile_requests": len(queries), "lexical_index_ms": lexical_ms,
                 "embedding_index_ms": 1000 * (perf_counter() - start) if vectors is not None else None,
                 "modes": {mode: measure(lambda batch: search(batch, mode), queries) for mode in modes},
                 "process_high_water_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
        output["datasets"][dataset] = entry
        write_json(args.output, output)
        print(dataset, entry, flush=True)


if __name__ == "__main__":
    main()
