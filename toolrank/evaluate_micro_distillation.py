"""Compare both student objectives on the family-disjoint development catalog."""
import argparse
from collections import defaultdict
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import torch

from examples.catalog_lexical import CatalogLexicalIndex, top_indices
from toolrank.baseline_comparison import measure, paired_interval
from toolrank.core import BGE, MODELS, load_model, read_json, write_json
from toolrank.fetch import REVISIONS
from toolrank.train_micro_distillation import PROTOCOL, digest


def summarize(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def evaluate(model, docs, questions, judgments, families, weights, batch):
    with torch.inference_mode():
        model.eval()
        tool_vectors = model.encode_document(docs, batch_size=batch, normalize_embeddings=True, convert_to_tensor=True)
        query_vectors = model.encode_query(questions, batch_size=batch, normalize_embeddings=True, convert_to_tensor=True)
        cosine = (query_vectors.float() @ tool_vectors.float().T).cpu().numpy()
    rows = {}
    for ranker, scores in (("dense", cosine), ("hybrid", cosine + weights)):
        by_family = defaultdict(list)
        queries = []
        for score, positives, family in zip(scores, judgments, families, strict=True):
            result = measure(top_indices(score, 10).tolist(), positives)
            queries.append(result)
            by_family[family].append(result)
        group_metrics = {family: summarize(metrics) for family, metrics in sorted(by_family.items())}
        rows[ranker] = {"query_weighted": summarize(queries), "macro": summarize(list(group_metrics.values())),
                        "families": group_metrics}
    return rows


def compare(candidate, baseline, seed):
    families = sorted(candidate["families"])
    if families != sorted(baseline["families"]):
        raise ValueError("Expected identical paired development families")
    return paired_interval([candidate["families"][f]["ndcg@10"] for f in families],
                           [baseline["families"][f]["ndcg@10"] for f in families], families, seed=seed)


def run(args):
    protocol = read_json(PROTOCOL)
    if digest(args.dev) != protocol["dev_sha256"]:
        raise ValueError("Expected the audited development split")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(.25)
    dataset = read_json(args.dev)
    tool_ids = sorted(dataset["tools"])
    docs = [f"{dataset['tools'][key]['name']}: {dataset['tools'][key]['description']}" for key in tool_ids]
    index = {key: i for i, key in enumerate(tool_ids)}
    questions = [row["query"] for row in dataset["queries"]]
    judgments = [dict.fromkeys((index[key] for key in row["positives"]), 1) for row in dataset["queries"]]
    families = ["+".join(row["families"]) for row in dataset["queries"]]
    lexical = CatalogLexicalIndex(docs)
    tfidf, bm25 = np.stack([lexical.scores(query) for query in questions]).transpose(1, 0, 2)
    tfidf_weight, bm25_weight = protocol["development"]["hybrid_weights"]
    weights = tfidf_weight * tfidf + bm25_weight * bm25
    training = read_json(args.training_record)
    if training["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Training protocol mismatch")
    models = {"micro": args.student, "toolrank": args.teacher, "bge-small": MODELS / BGE}
    models.update({name: args.models / name for name in training["models"]})
    expected = {name: row["weights_sha256"] for name, row in training["models"].items()}
    expected.update({"micro": protocol["student"]["weights_sha256"], "toolrank": protocol["teacher"]["weights_sha256"]})
    if len(training["models"]) != 2 * len(protocol["seeds"]):
        raise ValueError("Expected both arms for every registered seed")
    results = {}
    for name, root in models.items():
        checkpoint_hash = digest(root / "model.safetensors")
        if name in expected and expected[name] != checkpoint_hash:
            raise ValueError(f"Checkpoint mismatch for {name}")
        # The named BGE load applies its standard query prompt.
        model = load_model(BGE if name == "bge-small" else str(root.resolve()), args.device)
        model.max_seq_length = protocol["training"]["max_sequence_length"]
        started = perf_counter()
        results[name] = evaluate(model, docs, questions, judgments, families, weights, 64)
        results[name].update(weights_sha256=checkpoint_hash, parameters=sum(p.numel() for p in model.parameters()),
                             seconds=perf_counter() - started)
        del model
        print(name, results[name]["hybrid"]["macro"]["ndcg@10"], flush=True)
    paired = {}
    for seed in protocol["seeds"]:
        distilled = results[f"distilled-{seed}"]["hybrid"]
        labeled = results[f"labels-{seed}"]["hybrid"]
        paired[str(seed)] = {"distilled_minus_labels": compare(distilled, labeled, seed),
                             "distilled_minus_micro": compare(distilled, results["micro"]["hybrid"], seed),
                             "labels_minus_micro": compare(labeled, results["micro"]["hybrid"], seed)}
    gains = [row["distilled_minus_labels"]["delta"] for row in paired.values()]
    micro = [row["distilled_minus_micro"]["delta"] for row in paired.values()]
    settings = protocol["development"]
    decision = float(np.mean(gains)) >= settings["minimum_mean_macro_gain_over_labels"] and (
        min(micro) >= settings["minimum_seed_macro_gain_over_untuned_micro"])
    return {"development_only": True, "dev_sha256": digest(args.dev), "protocol_sha256": digest(PROTOCOL),
            "training_record_sha256": digest(args.training_record), "source_sha256": digest(__file__),
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "baseline_bge_revision": REVISIONS[f"models/{BGE}"], "device": args.device,
            "queries": len(questions), "tools": len(docs), "families": len(set(families)),
            "results": results, "paired_hybrid_macro": paired, "passes_development_screen": decision,
            "external_compute_usd": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("student", "teacher", "dev", "models", "training-record", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    args = parser.parse_args()
    write_json(args.output, run(args))
