"""Compare family-weighted and uniform micro training on development only."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import torch

from examples.catalog_lexical import CatalogLexicalIndex
from toolrank.audit_micro_errors import representations
from toolrank.baseline_comparison import paired_interval
from toolrank.core import load_model, read_json, write_json
from toolrank.evaluate_micro_distillation import evaluate
from toolrank.train_micro_distillation import PROTOCOL as PARENT
from toolrank.train_micro_distillation import digest, keys_and_judgments, teacher_scores, train_variant

PROTOCOL = Path(__file__).with_name("family_weighting_protocol.json")
METRICS = ("ndcg@10", "hit@1", "hit@5")


def family_weights(rows, exponent, cap):
    families = [tuple(row["families"]) for row in rows]
    frequencies = Counter(families)
    weights = torch.tensor([frequencies[family] ** -exponent for family in families], dtype=torch.float32)
    weights = (weights / weights.mean()).clamp(max=cap)
    return weights / weights.mean()


def evaluate_catalog(models, dataset, parent, args):
    docs, questions, relevant, _ = representations(dataset)
    grades = [dict.fromkeys(row, 1) for row in relevant]
    families = ["+".join(row["families"]) for row in dataset["queries"]]
    lexical = CatalogLexicalIndex(docs)
    tfidf, bm25 = np.stack([lexical.scores(query) for query in questions]).transpose(1, 0, 2)
    a, b = parent["development"]["hybrid_weights"]
    results = {}
    for name, path in models.items():
        model = load_model(str(path.resolve()), args.device)
        model.max_seq_length = parent["training"]["max_sequence_length"]
        metrics = evaluate(model, docs, questions, grades, families, a * tfidf + b * bm25, batch=64)
        results[name] = metrics["hybrid"]
    return {"tools": len(docs), "queries": len(questions), "families": len(set(families)), "hybrid": results}


def matched_differences(catalog, seeds):
    models = catalog["hybrid"]
    differences = {}
    for seed in seeds:
        uniform, weighted = (models[f"{name}-{seed}"] for name in ("control", "weighted"))
        families = sorted(uniform["families"])
        if families != sorted(weighted["families"]):
            raise ValueError("Expected the same paired development families")
        differences[str(seed)] = {
            metric: paired_interval([weighted["families"][name][metric] for name in families],
                                    [uniform["families"][name][metric] for name in families], families, seed=seed)
            for metric in METRICS
        }
    return differences


def gate_result(catalogs, gate):
    def deltas(catalog, metric):
        return [seed[metric]["delta"] for seed in catalogs[catalog]["weighted_minus_control"].values()]

    expanded_hit5 = deltas("expanded", "hit@5")
    outcomes = {
        "mean_expanded_macro_hit5_gain": float(np.mean(expanded_hit5)),
        "mean_expanded_macro_ndcg10_gain": float(np.mean(deltas("expanded", "ndcg@10"))),
        "mean_compact_macro_ndcg10_gain": float(np.mean(deltas("compact", "ndcg@10"))),
        "minimum_expanded_macro_hit5_gain": min(expanded_hit5),
    }
    checks = {
        "expanded_hit5": outcomes["mean_expanded_macro_hit5_gain"] >= gate["min_mean_expanded_macro_hit5_gain"],
        "expanded_ndcg": outcomes["mean_expanded_macro_ndcg10_gain"] >= gate["min_mean_expanded_macro_ndcg10_gain"],
        "compact_ndcg": outcomes["mean_compact_macro_ndcg10_gain"] >= -gate["max_mean_compact_macro_ndcg10_loss"],
        "seed_hit5": outcomes["minimum_expanded_macro_hit5_gain"] >= gate["min_expanded_macro_hit5_gain_per_seed"],
    }
    return {"outcomes": outcomes, "checks": checks, "passes": all(checks.values())}


def run(args):
    parent, protocol = read_json(PARENT), read_json(PROTOCOL)
    if digest(PARENT) != protocol["parent_protocol_sha256"]:
        raise ValueError("Training settings differ from the registered parent protocol")
    for item, path in (("train", args.train), ("dev", args.dev)):
        if digest(path) != protocol[f"{item}_sha256"]:
            raise ValueError(f"Unexpected {item} split")
    for name, path in (("student", args.student), ("teacher", args.teacher)):
        if digest(path / "model.safetensors") != parent[name]["weights_sha256"]:
            raise ValueError(f"Unexpected {name} checkpoint")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(.25)

    training, dev = read_json(args.train), read_json(args.dev)
    if training["tools"].keys() & dev["tools"].keys():
        raise ValueError("Expected disjoint original training and development tool IDs")
    if {tuple(row["families"]) for row in training["queries"]} & {tuple(row["families"]) for row in dev["queries"]}:
        raise ValueError("Expected disjoint training and development families")
    documents, questions, _, _ = representations(training)
    _, positive, allowed = keys_and_judgments(training)
    config = parent["training"]
    teacher = teacher_scores(args.teacher, questions, documents, args.device, config["max_sequence_length"])
    negatives = allowed & ~positive
    if not negatives.any(dim=1).all():
        raise ValueError("Each query needs an allowed hard negative")
    hard = teacher.masked_fill(~negatives, float("-inf")).argmax(dim=1)
    inputs = (questions, documents, positive, allowed, teacher, hard)
    weighting = protocol["weighting"]
    weights = family_weights(training["queries"], weighting["inverse_frequency_exponent"],
                             weighting["pre_normalization_cap_to_mean"])
    started = perf_counter()
    report = {
        "protocol": protocol, "protocol_sha256": digest(PROTOCOL), "source_sha256": digest(__file__),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__, "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "training": {"queries": len(questions), "tools": len(documents),
                     "families": len({tuple(row["families"]) for row in training["queries"]}),
                     "minimum_weight": float(weights.min()), "maximum_weight": float(weights.max()),
                     "checkpoints": {name: parent[name] for name in ("student", "teacher")}, "models": {}},
        "external_compute_usd": 0,
    }
    models = {}
    model_root = args.output
    for seed in protocol["seeds"]:
        for arm, row_weights in (("control", None), ("weighted", weights)):
            args.output = model_root / arm
            name = f"{arm}-{seed}"
            model_dir = args.output / f"labels-{seed}"
            if model_dir.exists():
                raise FileExistsError(model_dir)
            record = train_variant(args, config, seed, "labels", inputs, row_weights=row_weights)
            models[name] = model_dir
            report["training"]["models"][name] = record
            write_json(args.record, report)
    models["toolrank-33m"] = args.teacher
    spectra = {"compact": dev, "expanded": {"tools": dev["tools"] | training["tools"], "queries": dev["queries"]}}
    catalogs = {}
    for name, dataset in spectra.items():
        catalogs[name] = evaluate_catalog(models, dataset, parent, args)
        catalogs[name]["weighted_minus_control"] = matched_differences(catalogs[name], protocol["seeds"])
    report["development"] = catalogs
    report["gate"] = gate_result(catalogs, protocol["development_gate"])
    report["wall_seconds"] = perf_counter() - started
    write_json(args.record, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("student", "teacher", "train", "dev", "output", "record"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run(parser.parse_args())
