#!/usr/bin/env bash
set -euo pipefail
python_bin="${PYTHON:-.venv/bin/python}"
"$python_bin" - <<'PY'
from statistics import mean
from toolrank.core import read_json, write_json
target = read_json("results/mcp-modern-split.json")
baseline = read_json("results/etom/micro-base-dev.json")
train = target["splits"]["train"]["sha256"]
dev = target["splits"]["dev"]["sha256"]
assert baseline["corpus_sha256"] == dev
scores = {}
for seed in (11, 23, 37):
    model = f"micro-modern-{seed}"
    record = read_json(f"results/training/{model}.json")
    result = read_json(f"results/etom/{model}-dev.json")
    if record["training_sha256"] != train or result["corpus_sha256"] != dev:
        raise ValueError(f"corpus mismatch for {model}")
    scores[str(seed)] = result["metrics"]["ndcg@10"]
base_score = baseline["metrics"]["ndcg@10"]
if min(scores.values()) <= base_score:
    raise ValueError("the micro recipe did not improve over its base in every development seed")
write_json("results/micro-selection.json", {
    "selected": "artifacts/micro-modern-11", "predeclared_seed": 11,
    "base_ndcg@10": base_score, "mean_ndcg@10": mean(scores.values()), "seed_ndcg@10": scores,
})
PY
for schema in mcp plain openai; do
    for spec in 'micro-base TaylorAI/bge-micro-v2' 'micro artifacts/micro-modern-11'; do
        read -r name model <<< "$spec"
        calibration=".work/next/etom/$name-$schema-calibration.json"
        "$python_bin" -m toolrank.benchmark --split dev --schema "$schema" --model "$model" \
            --max-length 512 --skip-profile --output ".work/next/etom/$name-$schema-calibration-metrics.json" \
            --rows "$calibration"
        cutoff=$("$python_bin" -m toolrank.calibrate --threshold "$calibration")
        "$python_bin" -m toolrank.benchmark --split etom --schema "$schema" --model "$model" \
            --max-length 512 --batch 32 --skip-profile --threshold "$cutoff" \
            --output "results/etom/$name-$schema.json" --rows ".work/next/etom/$name-$schema-rows.json"
    done
    "$python_bin" -m toolrank.etom_compare --schema "$schema" --candidate micro \
        --reference micro-base --reference bge --reference scout --reference candidate --reference rex \
        --output "results/etom/micro-$schema-paired.json"
done
