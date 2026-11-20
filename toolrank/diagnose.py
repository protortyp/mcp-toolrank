from collections import defaultdict

import numpy as np

from toolrank.benchmark import toolret
from toolrank.core import BGE, load_model, read_json, write_json


def run():
    docs, queries = toolret()
    ids = sorted(docs)
    tokenizer = load_model(BGE).tokenizer
    cutoff = read_json("results/candidate-toolret.json")["max_seq_length"]
    long_docs = set()
    for i in range(0, len(ids), 512):
        chunk = ids[i:i + 512]
        sizes = tokenizer([docs[doc] for doc in chunk], truncation=True,
                          max_length=cutoff + 1, return_length=True)["length"]
        long_docs.update(doc for doc, size in zip(chunk, sizes) if size > cutoff)
    candidate = read_json(".work/candidate-toolret-rows.json")
    scout = {row["id"]: row for row in read_json(".work/toolscout-bge-small-toolret-rows.json")}
    judgments = {query_id: gold for query_id, _, _, gold in queries}
    buckets = defaultdict(list)
    for row in candidate:
        if not row["answerable"]:
            continue
        key = row["id"]
        long_positive = any(doc in long_docs for doc, relevance in judgments[key].items() if relevance > 0)
        label = "one_or_more_positive_documents_over_limit" if long_positive else "all_positive_documents_within_limit"
        buckets[label].append((row["metrics"]["ndcg@10"], scout[key]["metrics"]["ndcg@10"]))
    output = {"candidate_token_limit": cutoff, "documents_over_limit": len(long_docs), "total_documents": len(docs)}
    for group, scores in buckets.items():
        output[group] = {
            "queries": len(scores),
            "candidate_ndcg@10": float(np.mean([a for a, _ in scores])),
            "scout_ndcg@10": float(np.mean([b for _, b in scores])),
        }
    write_json("results/toolret-diagnostics.json", output)


if __name__ == "__main__":
    run()
