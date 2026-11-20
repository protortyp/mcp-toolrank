"""Reproduce standalone NumPy and package lexical agreement on frozen catalogs."""
import argparse
from pathlib import Path
from statistics import mean
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

from examples.benchmark_catalog_index import fetch_evaluation
from examples.catalog_lexical import CatalogLexicalIndex
from examples.rank_onnx import top_indices
from toolrank.core import load_model, metrics, write_json
from toolrank.lexical import Bm25Index, lexical_text
from toolrank.replay_hybrid import corpus

def compare(model, dataset):
    docs, rows = corpus(dataset, Path(".work/next/etom"))
    keys = sorted(docs)
    documents = [docs[key] for key in keys]
    cases = [(text, {key: grade for key, grade in gold.items() if key in docs and grade > 0})
             for _, text, _, gold in rows]
    queries = [query[0] for query in cases]
    options = dict(batch_size=64, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    doc_vectors = model.encode_document(documents, **options)
    query_vectors = model.encode_query(queries, **options)
    standalone = CatalogLexicalIndex(documents)
    vectorizer = TfidfVectorizer(preprocessor=lexical_text, ngram_range=(1, 2))
    tfidf_docs = vectorizer.fit_transform(documents)
    tfidf_query = vectorizer.transform(queries)
    bm25 = Bm25Index(documents)
    differences = [0.0, 0.0]
    top1 = top10 = 0
    totals = {"standalone": [], "package": []}
    for offset in range(0, len(cases), 32):
        batch = slice(offset, offset + 32)
        dense = (query_vectors[batch] @ doc_vectors.T).cpu().numpy()
        tfidf_gold = (tfidf_query[batch] @ tfidf_docs.T).toarray()
        bm25_gold = bm25.scores(queries[batch])
        for row, ((query, gold), vector) in enumerate(zip(cases[batch], dense, strict=True)):
            terms, names = standalone.scores(query)
            differences[0] = max(differences[0], float(np.abs(terms - tfidf_gold[row]).max()))
            differences[1] = max(differences[1], float(np.abs(names - bm25_gold[row]).max()))
            baseline = top_indices(vector + .4 * tfidf_gold[row] + .15 * bm25_gold[row], 10)
            actual = top_indices(vector + .4 * terms + .15 * names, 10)
            top1 += int(baseline[0] == actual[0])
            top10 += int(np.array_equal(baseline, actual))
            if gold:
                for name, ranked in (("standalone", actual), ("package", baseline)):
                    totals[name].append(metrics([keys[index] for index in ranked], gold))
    for query in queries[:16]:
        standalone.scores(query)
    timings = []
    for query in queries[:384]:
        start = time.perf_counter()
        standalone.scores(query)
        timings.append(1000 * (time.perf_counter() - start))
    return {
        "tools": len(keys), "queries": len(queries),
        "max_tfidf_absolute_difference": differences[0],
        "max_bm25_absolute_difference": differences[1],
        "top1_identical": top1, "ordered_top10_identical": top10,
        "standalone_tfidf_and_bm25_cpu_p50_ms": float(np.median(timings)),
        "metrics": {name: {key: mean(row[key] for row in results) for key in results[0]}
                    for name, results in totals.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--fetch-evaluation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fetch_evaluation:
        fetch_evaluation()
    torch.set_num_threads(4)
    model = load_model(str(args.checkpoint), args.device)
    results = {}
    for name in ("etom-plain", "etom-mcp", "api", "toolret"):
        results[name] = compare(model, name)
        write_json(args.output, results)
        print(name, results[name]["top1_identical"], results[name]["ordered_top10_identical"], flush=True)


if __name__ == "__main__":
    main()
