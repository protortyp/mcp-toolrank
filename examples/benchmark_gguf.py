"""Compare a local llama.cpp ToolRank endpoint with SentenceTransformers."""

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from urllib.request import Request, urlopen

import numpy as np
from sklearn.metrics import roc_auc_score
from tokenizers import Tokenizer
import torch

from toolrank.core import load_model, metrics, read_json, write_json
from toolrank.replay_hybrid import corpus

def make_encoder(url, tokenizer):
    endpoint = url.rstrip("/") + "/v1/embeddings"

    def submit(values):
        request = Request(endpoint, data=json.dumps({"input": values}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            rows = sorted(json.load(response)["data"], key=lambda row: row["index"])
        return np.array([row["embedding"] for row in rows], dtype=np.float32)

    def encode(texts):
        batches = []
        for offset in range(0, len(texts), 16):
            chunk = texts[offset:offset + 16]
            tokenized = tokenizer.encode_batch(chunk)
            long = [len(item.ids) > 512 for item in tokenized]
            if any(long):
                # For over-length inputs, preserve CLS, SEP, and first 510 content tokens.
                vectors = np.concatenate([submit(item.ids[:511] + item.ids[-1:] if truncated else text)
                                          for item, text, truncated in zip(tokenized, chunk, long, strict=True)])
            else:
                vectors = submit(chunk)
            assert vectors.shape == (len(chunk), 384)
            assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
            batches.append(vectors)
        return np.concatenate(batches)

    return encode


def results(scores, ids, rows):
    top = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    records, answerable = [], []
    available = set(ids)
    for ranks, (_, _, _, gold) in zip(top, rows, strict=True):
        relevant = {key: grade for key, grade in gold.items() if key in available and grade > 0}
        answerable.append(bool(relevant))
        if relevant:
            records.append(metrics([ids[index] for index in ranks], relevant))
    record = {key: mean(row[key] for row in records) for key in records[0]}
    if not all(answerable):
        record["answerability_auroc"] = float(roc_auc_score(answerable, scores.max(axis=1)))
    return record, top


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="local SentenceTransformer checkpoint")
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--etom-root", type=Path, default=Path(".work/next/etom"))
    parser.add_argument("--datasets", nargs="+", choices=("api", "etom-plain"), default=("api", "etom-plain"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    checkpoint = args.checkpoint
    model = load_model(str(checkpoint), args.device)
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    serve = make_encoder(args.url, tokenizer)
    prompt = model.prompts["query"]
    opts = dict(normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    outputs = {}
    for dataset in args.datasets:
        docs, rows = corpus(dataset, args.etom_root)
        ids = sorted(docs)
        qtexts = [row[1] for row in rows]
        dtexts = [docs[key] for key in ids]
        gold_q = model.encode_query(qtexts, **opts)
        gold_d = model.encode_document(dtexts, **opts)
        obtained_q = serve([prompt + text for text in qtexts])
        obtained_d = serve(dtexts)
        reference, ref_ranks = results(gold_q @ gold_d.T, ids, rows)
        observed, obs_ranks = results(obtained_q @ obtained_d.T, ids, rows)
        outputs[dataset] = {
            "queries": len(rows), "catalog": len(docs),
            "sentence_transformers": reference, "llama_cpp": observed,
            "identical_top10_order": int((ref_ranks == obs_ranks).all(axis=1).sum()),
            "mean_query_cosine_vs_reference": float((gold_q * obtained_q).sum(axis=1).mean()),
            "min_query_cosine_vs_reference": float((gold_q * obtained_q).sum(axis=1).min()),
        }
        write_json(args.output, outputs)
        print(dataset, reference["ndcg@10"], observed["ndcg@10"], flush=True)
    timings = []
    cases = read_json(Path("data/performance.json"))
    for query in cases:
        serve([prompt + query])
    for _ in range(4):
        for query in cases:
            start = perf_counter()
            serve([prompt + query])
            timings.append(perf_counter() - start)
    outputs["http_query_latency"] = {
        "p50_ms": float(np.median(timings)) * 1000,
        "p95_ms": float(np.quantile(timings, 0.95)) * 1000,
        "queries": len(timings),
        "includes_http_and_tokenization": True,
    }
    write_json(args.output, outputs)


if __name__ == "__main__":
    main()
