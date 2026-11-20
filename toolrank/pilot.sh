#!/usr/bin/env bash
set -euo pipefail
python_bin="${PYTHON:-.venv/bin/python}"
for recipe in inbatch family mined; do
    case "$recipe" in
        inbatch) options=(--batch 16 --lr 2e-5 --temperature 0.05) ;;
        family) options=(--batch 8 --lr 1e-5 --temperature 0.05) ;;
        mined) options=(--batch 16 --lr 2e-5 --temperature 0.08) ;;
    esac
    for seed in 11 23 37; do
        name="$recipe-$seed"
        "$python_bin" -m toolrank.train --seed "$seed" --negatives "$recipe" "${options[@]}" \
            --output "artifacts/$name" --record "results/training/$name.json"
        "$python_bin" -m toolrank.benchmark --model "artifacts/$name" --split dev \
            --output "results/dev/$name.json" --rows ".work/$name-dev-rows.json"
    done
done
