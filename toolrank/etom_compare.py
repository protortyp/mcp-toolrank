"""Compare fixed checkpoints on ETOM using paired provider-cluster intervals."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from toolrank.core import read_json, write_json
from toolrank.etom import load


def paired(candidate, baseline, schema):
    _, queries, _ = load(schema)
    cluster = {key: sorted(name.split("::")[0] for name in gold)[0]
               for key, _, _, gold in queries if gold}
    observed = {row["id"]: row["metrics"] for row in candidate if row["answerable"]}
    control = {row["id"]: row["metrics"] for row in baseline if row["answerable"]}
    if observed.keys() != control.keys() or observed.keys() != cluster.keys():
        raise ValueError("ETOM comparisons require identical positive queries")
    grouped = defaultdict(list)
    for key in sorted(observed):
        grouped[cluster[key]].append([observed[key][metric] - control[key][metric] for metric in ("hit@1", "ndcg@10")])
    sums = np.array([np.sum(values, axis=0) for values in grouped.values()])
    counts = np.array([len(values) for values in grouped.values()])
    rng = np.random.default_rng(418)
    samples = rng.integers(0, len(counts), size=(10000, len(counts)))
    simulated = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)[:, None]
    intervals = np.quantile(simulated, [0.025, 0.975], axis=0)
    deltas = sums.sum(axis=0) / counts.sum()
    return {"queries": len(observed), "provider_clusters": len(counts),
            "effects": {name: {"delta": float(deltas[i]), "95_interval": intervals[:, i].tolist()}
                        for i, name in enumerate(("hit@1", "ndcg@10"))}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", choices=["plain", "mcp", "openai"], default="mcp")
    parser.add_argument("--rows", type=Path, default=Path(".work/next/etom"))
    parser.add_argument("--candidate", default="candidate")
    parser.add_argument("--reference", action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    candidate = read_json(args.rows / f"{args.candidate}-{args.schema}-rows.json")
    comparisons = {}
    for reference in args.reference or ("bge", "scout", "rex"):
        baseline = read_json(args.rows / f"{reference}-{args.schema}-rows.json")
        comparisons[reference] = paired(candidate, baseline, args.schema)
    write_json(args.output or f"results/etom/{args.schema}-paired.json", comparisons)
