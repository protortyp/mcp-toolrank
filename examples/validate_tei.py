import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import torch

from toolrank.core import load_model

DOCS = [
    "list_exhibits: List exhibits at the museum.",
    "schedule_tour: Reserve a guided museum tour.",
    "cancel_tour: Remove a guided tour reservation.",
    "current_weather: Get the current temperature and weather.",
    "find_book: Find a title in the library catalog.",
    "renew_book: Extend the due date on library loans.",
    "retrieve_invoice: Download a copy of an invoice.",
    "list_transactions: Show previous transactions.",
    "long_exhibit: Exhibition ceramics; " + "collection " * 650,
]
QUERIES = [
    "What exhibitions can I see today?",
    "Can you schedule a museum tour?",
    "I need to cancel the tour I booked.",
    "What temperature is it outside?",
    "Find a book in the library.",
    "I want to keep a borrowed book for another week.",
    "Please get a copy of my bill.",
    "Show me my payment history.",
    "Exhibit collection " + "background " * 600,
]


def post(base, payload):
    request = Request(base + "/embed", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        return np.array(json.load(response), dtype=np.float32)


def stats(left, right):
    return {"min_dot_product": float((left * right).sum(axis=1).min()),
            "max_absolute_error": float(np.abs(left - right).max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = load_model(args.model, "cpu")
    reference_docs = model.encode_document(DOCS, normalize_embeddings=True)
    reference_queries = model.encode_query(QUERIES, normalize_embeddings=True)
    options = {"normalize": True, "truncate": True}
    docs = post(args.url, {"inputs": DOCS, **options})
    queries = post(args.url, {"inputs": QUERIES, "prompt_name": "query", **options})
    no_prompt = post(args.url, {"inputs": QUERIES, **options})
    assert docs.shape == queries.shape == (9, 384)
    assert np.linalg.norm(no_prompt - queries, axis=1).mean() > .01
    reference = reference_queries @ reference_docs.T
    served = queries @ docs.T
    rankings_match = np.array_equal(np.argsort(-reference, axis=1)[:, :3], np.argsort(-served, axis=1)[:, :3])
    record = {"documents": stats(reference_docs, docs), "queries": stats(reference_queries, queries),
              "ordered_top3_match": bool(rankings_match), "maximum_score_error": float(np.abs(reference - served).max()),
              "sample_count": len(QUERIES) + len(DOCS)}
    with urlopen(args.url + "/info", timeout=10) as response:
        record["server"] = json.load(response)
    assert record["maximum_score_error"] < 1e-4 and rankings_match, record
    assert record["server"]["max_input_length"] == 512
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
