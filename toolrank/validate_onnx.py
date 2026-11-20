"""Compare frozen retrieval and rejection metrics for both ToolRank CPU exports."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from toolrank.benchmark import local_corpus
from toolrank.core import metrics, read_json, write_json
from toolrank.etom import load


def score(session, tokenizer, prefix, docs, queries):
    keys = sorted(docs)
    names = [item.name for item in session.get_inputs()]

    def encode(items):
        chunks = []
        for start in range(0, len(items), 24):
            tokens = tokenizer(items[start:start + 24], padding=True, truncation=True, max_length=512,
                               return_tensors="np")
            chunks.append(session.run(None, {name: tokens[name] for name in names})[0])
        vectors = np.concatenate(chunks)
        assert np.isfinite(vectors).all() and np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1)) < 1e-4
        return vectors

    query_vectors = encode([prefix + query for _, query, _, _ in queries])
    doc_vectors = encode([docs[key] for key in keys])
    similarities = query_vectors @ doc_vectors.T
    ranks = np.argsort(-similarities, axis=1, kind="stable")[:, :10]
    rows = []
    for i, ((query_id, _, family, gold), indices) in enumerate(zip(queries, ranks, strict=True)):
        ranking = [keys[j] for j in indices]
        row = {"id": query_id, "family": family, "max_score": float(similarities[i, indices[0]]),
               "answerable": bool(gold)}
        if gold:
            row["metrics"] = metrics(ranking, gold)
            row["metrics"]["hit@1"] = float(ranking[0] in gold)
        rows.append(row)
    return rows


def calibrate(rows):
    scores = [row["max_score"] for row in rows]

    def quality(threshold):
        coverage = np.mean([row["max_score"] >= threshold for row in rows if row["answerable"]])
        rejection = np.mean([row["max_score"] < threshold for row in rows if not row["answerable"]])
        return (coverage + rejection) / 2, rejection

    return float(max(scores + [np.nextafter(max(scores), np.inf)], key=quality))


def summarize(rows, threshold):
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    quality = {key: float(np.mean([row["metrics"][key] for row in answerable]))
               for key in answerable[0]["metrics"]}
    return {
        "metrics": quality,
        "threshold": threshold,
        "coverage": float(np.mean([row["max_score"] >= threshold for row in answerable])),
        "false_accept_rate": float(np.mean([row["max_score"] >= threshold for row in unanswerable])),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
    }


def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    prefix = read_json(args.checkpoint / "config_sentence_transformers.json")["prompts"]["query"]
    fresh = read_json(args.holdout)
    docs = {key: f"{tool['name']}: {tool['description']}" for key, tool in fresh["tools"].items()}
    queries = [(row["id"], row["query"], row["family"], dict.fromkeys(row["positives"], 1))
               for row in fresh["queries"]]
    freeze = {"sha256": sha256(args.holdout.read_bytes()).hexdigest(), "documents": len(docs),
              "queries": len(queries), "unanswerable": sum(not row[3] for row in queries)}
    cal_docs, cal_queries = local_corpus("dev", "plain")
    etom_docs, etom_queries, etom_audit = load("plain")
    results = {}
    for name, graph in (("fp32", "embedding.onnx"), ("int8", "embedding-int8.onnx")):
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = 4
        session = ort.InferenceSession(str(args.checkpoint / "onnx" / graph), sess_options=settings,
                                       providers=["CPUExecutionProvider"])
        threshold = calibrate(score(session, tokenizer, prefix, cal_docs, cal_queries))
        results[name] = {
            "fresh": summarize(score(session, tokenizer, prefix, docs, queries), threshold),
            "etom": summarize(score(session, tokenizer, prefix, etom_docs, etom_queries), threshold),
        }
        print(f"{name} validated", flush=True)
    write_json(args.output, {"calibration": "authored v1 development split", "fresh_freeze": freeze,
                             "etom_audit": etom_audit, "results": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--holdout", type=Path, default=Path("data/onnx_int8_holdout.json"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
