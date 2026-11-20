"""Measure no-tool rejection with deployment thresholds chosen on development only."""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
import torch

from toolrank.baseline_comparison import ETOM_ROOT, Lexical, WEIGHTS, digest, fingerprint
from toolrank.benchmark import local_corpus
from toolrank.core import BGE, load_model, write_json
from toolrank.replay_hybrid import corpus


TARGET = .10
METHODS = ("max_cosine", "hybrid_max", "margin", "standardized_top")


def signals(cosine, hybrid):
    best = np.partition(cosine, -2, axis=1)[:, -2:]
    top = best[:, 1]
    return {
        "max_cosine": top,
        "hybrid_max": hybrid.max(axis=1),
        "margin": top - best[:, 0],
        "standardized_top": (top - cosine.mean(axis=1)) / np.maximum(cosine.std(axis=1), 1e-6),
    }


def cutoff(scores, supported):
    negative = scores[~supported]
    if not len(negative) or not supported.any():
        raise ValueError("calibration needs supported and unsupported requests")
    maximum_false_accepts = int(np.floor(TARGET * len(negative)))
    boundary = np.sort(negative)[::-1][maximum_false_accepts]
    return float(np.nextafter(np.float64(boundary), np.inf))


def metrics(scores, supported, correct, threshold):
    accepted = np.asarray(scores, dtype=np.float64) >= threshold
    true_accept = accepted & supported
    false_accept = accepted & ~supported
    negative = int((~supported).sum())
    errors = int(false_accept.sum())
    selective_accuracy = float(correct[accepted].mean()) if accepted.any() else None
    return {"threshold": float(threshold), "supported": int(supported.sum()), "unsupported": negative,
            "accepted_supported": int(true_accept.sum()), "accepted_unsupported": errors,
            "supported_coverage": float(accepted[supported].mean()),
            "false_acceptance": float(accepted[~supported].mean()),
            "false_acceptance_upper_95": 1.0 if errors == negative else float(beta.ppf(.95, errors + 1, negative - errors)),
            "selective_hit_1": selective_accuracy,
            "risk": None if selective_accuracy is None else 1 - selective_accuracy}


def score_catalog(model, documents, rows):
    keys = sorted(documents)
    texts = [documents[key] for key in keys]
    queries = [row[1] for row in rows]
    options = {"batch_size": 32, "normalize_embeddings": True, "show_progress_bar": False}
    cosine = model.encode_query(queries, **options) @ model.encode_document(texts, **options).T
    tfidf, bm25 = Lexical(texts).score(queries)
    hybrid = cosine + WEIGHTS["tfidf"] * tfidf + WEIGHTS["bm25"] * bm25
    positives = [{key for key, grade in row[3].items() if grade > 0} for row in rows]
    supported = np.array([bool(gold) for gold in positives])
    correct = np.array([keys[index] in gold for index, gold in zip(hybrid.argmax(axis=1), positives, strict=True)])
    return signals(cosine, hybrid), supported, correct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolrank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    datasets = ("etom-plain", "etom-mcp", "etom-openai", "api")
    output = {"target_development_false_acceptance": TARGET,
              "development_negative_count": 6,
              "development_file_sha256": digest(Path("data/dev.json")),
              "development_threshold_rule": "Smallest attainable threshold allowing at most floor(0.10*n) false accepts",
              "deployment_assumption": "All thresholds transfer from authored development. Sweep is descriptive only.",
              "device": args.device, "models": {}, "datasets": {name: fingerprint(name) for name in datasets}}
    for name, path in (("bge", BGE), ("toolrank", str(args.toolrank.resolve()))):
        model = load_model(path, args.device, {"dtype": torch.float32})
        model.max_seq_length = 512
        report = {}
        for dataset in datasets:
            schema = dataset.removeprefix("etom-") if dataset.startswith("etom") else "plain"
            dev, dev_supported, dev_correct = score_catalog(model, *local_corpus("dev", schema))
            tests, supported, correct = score_catalog(model, *corpus(dataset, ETOM_ROOT))
            entries = {}
            for method in METHODS:
                threshold = cutoff(dev[method], dev_supported)
                scores = tests[method]
                points = np.unique(np.r_[np.quantile(scores, np.linspace(0, 1, 21)),
                                         np.nextafter(np.float64(scores.max()), np.inf)])
                entries[method] = {
                    "development": metrics(dev[method], dev_supported, dev_correct, threshold),
                    "transfer": metrics(scores, supported, correct, threshold),
                    "roc_auc_answerability": float(roc_auc_score(supported, scores)),
                    "threshold_sweep": [metrics(scores, supported, correct, point) for point in points],
                }
            report[dataset] = entries
            print(name, dataset, {method: (round(entry["transfer"]["supported_coverage"], 3),
                                           round(entry["transfer"]["false_acceptance"], 3))
                                  for method, entry in entries.items()}, flush=True)
        output["models"][name] = report
        write_json(args.output, output)


if __name__ == "__main__":
    main()
