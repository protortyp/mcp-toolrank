"""Select one recipe using only the family-disjoint development measurements."""
from statistics import mean

from toolrank.core import read_json, write_json


SEEDS = (11, 23, 37)
RECIPES = {"initial": "modern", "conservative": "modern-conservative"}


if __name__ == "__main__":
    scores = {}
    corpus = read_json("results/mcp-modern-split.json")
    train_hash = corpus["splits"]["train"]["sha256"]
    dev_hash = corpus["splits"]["dev"]["sha256"]
    for recipe, prefix in RECIPES.items():
        per_seed = {}
        for seed in SEEDS:
            name = f"{prefix}-{seed}"
            train = read_json(f"results/training/{name}.json")
            result = read_json(f"results/etom/{name}-dev.json")
            if train["training_sha256"] != train_hash or result["corpus_sha256"] != dev_hash:
                raise ValueError(f"inconsistent training or development corpus for {name}")
            per_seed[str(seed)] = {key: result["metrics"][key] for key in ("ndcg@10", "recall@1")}
        scores[recipe] = {
            "per_seed": per_seed,
            "mean_ndcg@10": mean(row["ndcg@10"] for row in per_seed.values()),
            "mean_recall@1": mean(row["recall@1"] for row in per_seed.values()),
        }
    winner = max(scores, key=lambda key: (scores[key]["mean_ndcg@10"], scores[key]["mean_recall@1"], key == "conservative"))
    model = f"artifacts/{RECIPES[winner]}-11"
    write_json("results/modern-selection.json", {"selected": model, "recipe": winner, "development": scores})
    print(model)
