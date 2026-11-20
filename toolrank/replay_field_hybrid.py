"""Replay the field-aware BM25 hybrid on pinned ToolRank evaluation suites."""
import argparse
from pathlib import Path
from statistics import mean
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

from toolrank.core import load_model, metrics, write_json
from toolrank.replay_hybrid import corpus
from toolrank.lexical import Bm25Index, lexical_text, repeat_name


def evaluate(model, dataset):
    documents, rows = corpus(dataset, Path(".work/next/etom"))
    ids = sorted(documents)
    texts = [documents[key] for key in ids]
    judged = [(request, {key: grade for key, grade in gold.items() if key in documents and grade > 0})
              for _, request, _, gold in rows]
    queries = [request for request, _ in judged]
    vectorizer = TfidfVectorizer(preprocessor=lexical_text, ngram_range=(1, 2))
    lexical_documents = vectorizer.fit_transform(texts)
    lexical_queries = vectorizer.transform(queries)
    index = Bm25Index(texts)
    options = dict(batch_size=64, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    document_embeddings = model.encode_document(texts, **options)
    query_embeddings = model.encode_query(queries, **options)
    results = {"existing-hybrid": [], "candidate": []}
    for offset in range(0, len(queries), 64):
        batch = slice(offset, offset + 64)
        dense = (query_embeddings[batch] @ document_embeddings.T).cpu().numpy()
        lexical = (lexical_queries[batch] @ lexical_documents.T).toarray()
        baseline = dense + .4 * lexical
        candidate = baseline + .15 * index.scores(queries[batch])
        for name, scores in (("existing-hybrid", baseline), ("candidate", candidate)):
            ranked = np.argsort(-scores, axis=1, kind="stable")[:, :10]
            for (_, gold), ranks in zip(judged[batch], ranked, strict=True):
                if gold:
                    results[name].append(metrics([ids[index] for index in ranks], gold))

    for query in queries[:16]:
        index.scores([query])
    timings = []
    for query in queries[:384]:
        start = time.perf_counter()
        index.scores([query])
        timings.append(1000 * (time.perf_counter() - start))

    answerable = len(results["candidate"])
    return {
        "additional_bm25_cpu_p50_ms": float(np.median(timings)),
        "catalog": len(ids),
        "answerable": answerable,
        "queries_without_available_positive": len(rows) - answerable,
        "extra_name_field_count": sum(text != repeat_name(text) for text in texts),
        "metrics": {name: {key: mean(row[key] for row in records) for key in records[0]}
                    for name, records in results.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolrank", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = load_model(args.toolrank, args.device)
    results = {}
    for name in ("etom-plain", "etom-mcp", "api", "toolret"):
        result = evaluate(model, name)
        results[name] = result
        write_json(args.output, results)
        print(name, {label: round(value["ndcg@10"] * 100, 3)
                     for label, value in result["metrics"].items()}, flush=True)


if __name__ == "__main__":
    main()
