import argparse
from pathlib import Path

import numpy as np

from toolrank.core import read_json, write_json


RECIPES = ("inbatch", "family", "mined")
SEEDS = (11, 23, 37)


def choose():
    summary = {}
    for recipe in RECIPES:
        names = [f"{recipe}-{seed}" for seed in SEEDS]
        values = [read_json(f"results/dev/{name}.json")["metrics"]["recall@1"] for name in names]
        rate = read_json(f"results/training/{names[0]}.json")["config"]["lr"]
        summary[recipe] = {"mean_recall@1": float(np.mean(values)), "seed_recall@1": values, "lr": rate}
    winner = min(RECIPES, key=lambda recipe: (-summary[recipe]["mean_recall@1"], summary[recipe]["lr"]))
    return f"{winner}-{SEEDS[0]}", summary


def threshold(path):
    rows = read_json(path)
    scores = np.array([row["score"] for row in rows])
    labels = np.array([row["answerable"] for row in rows])
    if labels.all() or not labels.any():
        raise ValueError("threshold calibration requires both query classes")
    candidates = np.unique(np.r_[scores, np.nextafter(scores.max(), np.inf)])

    def evaluate(value):
        coverage = np.mean(scores[labels] >= value)
        rejection = np.mean(scores[~labels] < value)
        return (coverage + rejection) / 2, rejection

    return float(max(candidates, key=lambda value: (*evaluate(value), value)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=Path)
    args = parser.parse_args()
    if args.threshold:
        print(threshold(args.threshold))
    else:
        selected, recipes = choose()
        write_json("results/selection.json", {"selected": selected, "recipes": recipes})
        print(selected)
