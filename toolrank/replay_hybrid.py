"""Measure the same lexical weight across pinned tool benchmarks."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

from toolrank.benchmark import toolret
from toolrank.core import BGE, load_model, metrics, read_json, write_json
from toolrank.etom import load
from toolrank.selector import lexical_text

LEXICAL_WEIGHT = 0.4
BATCH_SIZE = 96
HOLDOUT = Path("data/onnx_int8_holdout.json")


def corpus(dataset, etom_root):
    if dataset == "toolret":
        return toolret()
    if dataset.startswith("etom-"):
        documents, queries, _ = load(dataset.removeprefix("etom-"), etom_root)
        return documents, queries

    data = read_json(HOLDOUT)
    documents = {
        key: f"{tool['name']}: {tool['description']}" for key, tool in data["tools"].items()
    }
    queries = [
        (row["id"], row["query"], row["family"], dict.fromkeys(row["positives"], 1))
        for row in data["queries"]
    ]
    return documents, queries


def evaluate(dataset, models, etom_root):
    documents, queries = corpus(dataset, etom_root)
    ids = sorted(documents)
    texts = [documents[key] for key in ids]
    judged = []
    for _, query, _, labels in queries:
        available = {key: grade for key, grade in labels.items() if key in documents and grade > 0}
        if available:
            judged.append((query, available))
    questions, gold = zip(*judged)

    vectorizer = TfidfVectorizer(preprocessor=lexical_text, ngram_range=(1, 2))
    lexical_documents = vectorizer.fit_transform(texts)
    lexical_queries = vectorizer.transform(questions)
    options = dict(batch_size=BATCH_SIZE, normalize_embeddings=True, convert_to_tensor=True,
                   show_progress_bar=False)
    vectors = {
        name: (model.encode_query(list(questions), **options), model.encode_document(texts, **options))
        for name, model in models.items()
    }
    totals = {"lexical": {}}
    for name in models:
        totals[f"{name}_dense"] = {}
        totals[f"{name}_hybrid"] = {}

    for offset in range(0, len(judged), BATCH_SIZE):
        chunk = slice(offset, offset + BATCH_SIZE)
        lexical = (lexical_queries[chunk] @ lexical_documents.T).toarray()
        scores = {"lexical": lexical}
        for name, (query_vectors, doc_vectors) in vectors.items():
            dense = (query_vectors[chunk] @ doc_vectors.T).cpu().numpy()
            scores[f"{name}_dense"] = dense
            scores[f"{name}_hybrid"] = dense + LEXICAL_WEIGHT * lexical
        for name, matrix in scores.items():
            ranked = np.argsort(-matrix, axis=1, kind="stable")[:, :10]
            for indices, labels in zip(ranked, gold[chunk], strict=True):
                for metric, value in metrics([ids[i] for i in indices], labels).items():
                    totals[name][metric] = totals[name].get(metric, 0) + value

    return {
        "documents": len(ids), "queries": len(judged),
        "excluded_without_available_positives": len(queries) - len(judged),
        "metrics": {
            name: {metric: value / len(judged) for metric, value in totals[name].items()}
            for name in totals
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolrank", required=True)
    parser.add_argument("--baseline", default=BGE)
    parser.add_argument("--dataset", nargs="+", default=["etom-plain", "etom-mcp", "api"],
                        choices=["etom-plain", "etom-mcp", "api", "toolret"])
    parser.add_argument("--etom-root", type=Path, default=Path(".work/next/etom"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    models = {
        "toolrank": load_model(args.toolrank, args.device),
        "bge": load_model(args.baseline, args.device),
    }
    results = {}
    for dataset in args.dataset:
        results[dataset] = evaluate(dataset, models, args.etom_root)
        print(dataset, {key: value["ndcg@10"] for key, value in results[dataset]["metrics"].items()},
              flush=True)
    write_json(args.output, {"lexical_weight": LEXICAL_WEIGHT, "device": args.device, "results": results})
