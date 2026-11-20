#!/usr/bin/env bash
set -euo pipefail
python_bin="${PYTHON:-.venv/bin/python}"
selected=$("$python_bin" -m toolrank.calibrate)
mkdir -p results/test results/calibration
for spec in 'bge BAAI/bge-small-en-v1.5' 'scout nuhmanpk/toolscout-bge-small' 'rex Lux1997/Tool-Embed-0.6B' "candidate artifacts/$selected"; do
    read -r name model <<< "$spec"
    "$python_bin" -m toolrank.benchmark --model "$model" --device cpu --profile-only \
        --output "results/$name-cpu.json"
    for schema in plain mcp openai; do
        rows=".work/$name-$schema-dev.json"
        "$python_bin" -m toolrank.benchmark --model "$model" --skip-profile --split dev --schema "$schema" --device cpu \
            --output "results/calibration/$name-$schema.json" --rows "$rows"
        threshold=$("$python_bin" -m toolrank.calibrate --threshold "$rows")
        "$python_bin" -m toolrank.benchmark --model "$model" --skip-profile --split test --schema "$schema" --device cpu \
            --threshold "$threshold" --output "results/test/$name-$schema.json" --rows "results/test/$name-$schema-rows.json"
    done
done
