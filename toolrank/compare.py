from collections import defaultdict
from pathlib import Path

import numpy as np

from toolrank.core import read_json, write_json


def paired(schema, reference):
    root = Path("results/test")
    candidate = {row["id"]: row for row in read_json(root / f"candidate-{schema}-rows.json")}
    baseline = {row["id"]: row for row in read_json(root / f"{reference}-{schema}-rows.json")}
    if candidate.keys() != baseline.keys():
        raise ValueError("comparison must use the same queries")
    families = defaultdict(list)
    misses = []
    for key, row in candidate.items():
        other = baseline[key]
        if not row["answerable"]:
            continue
        hit = row["metrics"]["recall@1"]
        families[row["family"]].append(hit - other["metrics"]["recall@1"])
        if hit != 1:
            misses.append({"query_id": key, "top1": row["top1"], "reference_top1": other["top1"]})
    values = np.array([np.mean(family) for family in families.values()])
    rng = np.random.default_rng(20261120)
    draws = rng.choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return {"family_macro_delta": float(values.mean()), "paired_cluster_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "families": len(families), "candidate_misses": misses}


if __name__ == "__main__":
    write_json("results/paired.json", {schema: {other: paired(schema, other) for other in ("bge", "scout", "rex")}
                                      for schema in ("plain", "mcp", "openai")})
