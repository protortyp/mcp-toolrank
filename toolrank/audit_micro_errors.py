"""Audit micro retriever errors and prioritize unjudged training pairs for review."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import torch

from examples.catalog_lexical import CatalogLexicalIndex, top_indices
from toolrank.baseline_comparison import paired_interval
from toolrank.core import PREFIX, load_model, read_json, write_json
from toolrank.train_micro_distillation import PROTOCOL, digest, keys_and_judgments


def representations(dataset):
    keys = sorted(dataset["tools"])
    tools = [dataset["tools"][key] for key in keys]
    documents = [f"{tool['name']}: {tool['description']}" for tool in tools]
    questions = [row["query"] for row in dataset["queries"]]
    key_to_position = {key: index for index, key in enumerate(keys)}
    relevant = [{key_to_position[key] for key in row["positives"]} for row in dataset["queries"]]
    repositories = [tool["repository"] for tool in tools]
    return documents, questions, relevant, repositories


def top1_category(index, relevant, repositories):
    if index in relevant:
        return "judged_positive"
    if any(repositories[index] == repositories[positive] for positive in relevant):
        return "unjudged_same_repository"
    return "unjudged_other_repository"


def judgment_outcomes(matrix, relevant, repositories):
    categories, hit5 = [], []
    for scores, gold in zip(matrix, relevant, strict=True):
        order = top_indices(scores, 5)
        categories.append(top1_category(order[0], gold, repositories))
        hit5.append(bool(gold.intersection(order)))
    return categories, hit5


def paired_misses(candidate, comparison):
    return {
        "both_miss": sum(not a and not b for a, b in zip(candidate, comparison, strict=True)),
        "comparison_only_hit": sum(not a and b for a, b in zip(candidate, comparison, strict=True)),
        "candidate_only_hit": sum(a and not b for a, b in zip(candidate, comparison, strict=True)),
        "both_hit": sum(a and b for a, b in zip(candidate, comparison, strict=True)),
    }


def encode_scores(model, documents, questions, *, length):
    model.max_seq_length = length
    with torch.inference_mode():
        model.eval()
        options = dict(batch_size=64, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        docs = model.encode_document(documents, **options)
        queries = model.encode_query(questions, **options)
        return (queries.float() @ docs.float().T).cpu().numpy()


def audit_mask(training, teacher_scores):
    keys, positive_tensor, allowed_tensor = keys_and_judgments(training)
    repositories = [training["tools"][key]["repository"] for key in keys]
    positive, allowed = positive_tensor.numpy(), allowed_tensor.numpy()
    if not (~positive).any(axis=1).all():
        raise ValueError("Expected at least one unjudged candidate for each training query")
    counts = Counter()
    pairs = Counter()
    queue = []
    for query, scores, judged, eligible in zip(training["queries"], teacher_scores, positive, allowed, strict=True):
        gold = np.flatnonzero(judged)
        best = gold[scores[gold].argmax()]
        unknown = np.flatnonzero(~judged)
        top = unknown[scores[unknown].argmax()]
        same_repository = {repositories[position] for position in gold}
        has_same_repo = any(repositories[position] in same_repository for position in unknown)
        counts["queries_with_unjudged_same_repository"] += has_same_repo
        counts["queries_with_masked_unknown"] += bool((~eligible).any())
        counts["top_unjudged_in_same_repository"] += repositories[top] in same_repository
        counts["top_unjudged_masked"] += not eligible[top]
        counts["top_unjudged_above_best_positive"] += bool(scores[top] > scores[best])
        if repositories[top] in same_repository:
            judged_siblings = [index for index in gold if repositories[index] == repositories[top]]
            anchor = judged_siblings[scores[judged_siblings].argmax()]
            pair = (keys[anchor], keys[top])
            pairs[pair] += 1
            counts["top_unjudged_same_repository_above_best_positive"] += bool(scores[top] > scores[best])
            queue.append({"query_id": query["id"], "judged_sibling": pair[0], "unjudged_candidate": pair[1],
                          "cosine_gap": float(scores[top] - scores[anchor])})
    queue.sort(key=lambda row: (-pairs[row["judged_sibling"], row["unjudged_candidate"]],
                                -row["cosine_gap"], row["query_id"]))
    return dict(counts) | {
        "queries": len(training["queries"]), "tools": len(keys), "repositories": len(set(repositories)),
        "distinct_ranked_pairs": len(pairs),
        "top_five_pairs_query_counts": sorted(pairs.values(), reverse=True)[:5],
    }, queue


class DevelopmentCatalog:
    def __init__(self, dataset, weights):
        self.docs, self.questions, self.relevant, self.repos = representations(dataset)
        self.families = ["+".join(row["families"]) for row in dataset["queries"]]
        lexical = CatalogLexicalIndex(self.docs)
        tfidf, bm25 = np.stack([lexical.scores(query) for query in self.questions]).transpose(1, 0, 2)
        self.blend = weights[0] * tfidf + weights[1] * bm25
        self.results = {}
        self.outcomes = {}

    def score(self, name, model, length):
        cosine = encode_scores(model, self.docs, self.questions, length=length)
        for ranker, matrix in (("dense", cosine), ("hybrid", cosine + self.blend)):
            label = f"{name}/{ranker}"
            categories, hit5 = judgment_outcomes(matrix, self.relevant, self.repos)
            self.outcomes[label] = ([category == "judged_positive" for category in categories], hit5)
            self.results[label] = {"top_one_judgments": dict(Counter(categories)), "hit_at_5": sum(hit5)}

    def report(self):
        comparisons = {}
        for comparison in ("toolrank/hybrid", "bge-small/hybrid", "micro/hybrid", "labels-11/dense"):
            comparisons[comparison] = {}
            for index, metric in enumerate(("hit_at_1", "hit_at_5")):
                student = self.outcomes["labels-11/hybrid"][index]
                other = self.outcomes[comparison][index]
                comparisons[comparison][metric] = paired_misses(student, other) | {
                    "student_minus_comparator": paired_interval(student, other, self.families),
                }
        return {"tools": len(self.docs), "queries": len(self.questions), "families": len(set(self.families)),
                "results": self.results, "paired_vs_student_hybrid": comparisons}


def run(args):
    previous = read_json(PROTOCOL)
    for split, path in (("train", args.train), ("dev", args.dev)):
        if digest(path) != previous[f"{split}_sha256"]:
            raise ValueError(f"Use the audited {split} split")
    baseline_path = Path(__file__).resolve().parents[1] / "results/micro_distillation_development.json"
    baseline = read_json(baseline_path)
    models = {"labels-11": args.labels, "toolrank": args.teacher, "bge-small": args.bge, "micro": args.micro}
    training, development = read_json(args.train), read_json(args.dev)
    if {tool["repository"] for tool in training["tools"].values()} & {
        tool["repository"] for tool in development["tools"].values()
    }:
        raise ValueError("Training and development repositories must be disjoint")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(.25)
    started = perf_counter()
    if training["tools"].keys() & development["tools"].keys():
        raise ValueError("Training and development tool IDs must be disjoint")
    weights = previous["development"]["hybrid_weights"]
    compact = DevelopmentCatalog(development, weights)
    expanded = DevelopmentCatalog({"tools": development["tools"] | training["tools"],
                                   "queries": development["queries"]}, weights)
    checksums = {}
    train_docs, train_queries, _, _ = representations(training)
    train_report = queue = None
    for name, path in models.items():
        checksums[name] = digest(path / "model.safetensors")
        if checksums[name] != baseline["results"][name]["weights_sha256"]:
            raise ValueError(f"Checkpoint mismatch for {name}")
        tool_model = load_model(str(path.resolve()), args.device)
        if name == "bge-small":
            tool_model.prompts["query"] = PREFIX
        length = previous["training"]["max_sequence_length"]
        compact.score(name, tool_model, length)
        expanded.score(name, tool_model, length)
        if name == "toolrank":
            train_matrix = encode_scores(tool_model, train_docs, train_queries, length=length)
            train_report, queue = audit_mask(training, train_matrix)
        del tool_model
    if args.review_queue is not None:
        # Local review aid. Identifiers refer to the audited training input only.
        args.review_queue.parent.mkdir(parents=True, exist_ok=True)
        with args.review_queue.open("x", encoding="utf-8") as stream:
            json.dump(queue, stream, indent=2)
            stream.write("\n")
    return {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": digest(__file__), "protocol_sha256": digest(PROTOCOL), "train_sha256": digest(args.train),
        "development_sha256": digest(args.dev), "checkpoints": checksums, "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "development": compact.report(), "development_with_training_distractors": expanded.report(),
        "training_mask": train_report, "external_compute_usd": 0, "wall_seconds": perf_counter() - started,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("train", "dev", "labels", "teacher", "bge", "micro", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--review-queue", type=Path, help="write local unjudged training pairs to a new JSON file")
    args = parser.parse_args()
    write_json(args.output, run(args))
