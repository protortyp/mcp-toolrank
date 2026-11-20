"""Reproduce ETOM rankings with NumPy search using the locked model runtime."""
import argparse

import numpy as np
import torch

from toolrank.core import load_model, metrics, write_json
from toolrank.etom import load


def run(model_id, schema, device):
    model = load_model(model_id, device)
    model.max_seq_length = 512
    docs, queries, audit = load(schema)
    ids = sorted(docs)
    docs_matrix = model.encode_document([docs[key] for key in ids], batch_size=32, normalize_embeddings=True)
    positives = [(query, gold) for _, query, _, gold in queries if gold]
    query_matrix = model.encode_query([query for query, _ in positives], batch_size=32, normalize_embeddings=True)
    scores = query_matrix @ docs_matrix.T
    top = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    per_query = []
    for ranked, (_, gold) in zip(top, positives, strict=True):
        ranking = [ids[i] for i in ranked]
        item = metrics(ranking, gold)
        item["hit@1"] = float(ranking[0] in gold)
        per_query.append(item)
    return {"model": model_id, "device": device, "schema": schema, "dataset_audit": audit,
            "metrics": {key: float(np.mean([row[key] for row in per_query])) for key in per_query[0]}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--schema", choices=("plain", "mcp", "openai"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.output, run(args.model, args.schema, args.device))
