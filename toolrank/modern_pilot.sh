#!/usr/bin/env bash
set -euo pipefail
python_bin="${PYTHON:-.venv/bin/python}"
dataset="${1:-.work/next/modern-expanded}"
for recipe in initial conservative; do
    case "$recipe" in
        initial) options=(--batch 24 --epochs 2 --lr 1e-5 --temperature 0.05) ;;
        conservative) options=(--batch 32 --epochs 1 --lr 5e-6 --temperature 0.05) ;;
    esac
    for seed in 11 23 37; do
        label="modern-$recipe-$seed"
        if [[ "$recipe" == initial ]]; then
            label="modern-$seed"
        fi
        "$python_bin" -m toolrank.train_modern --training "$dataset/train.json" --seed "$seed" "${options[@]}" \
            --output "artifacts/$label" --record "results/training/$label.json"
        "$python_bin" -m toolrank.benchmark --split modern-dev --corpus "$dataset/dev.json" \
            --model "artifacts/$label" --skip-profile --output "results/etom/$label-dev.json"
    done
done
