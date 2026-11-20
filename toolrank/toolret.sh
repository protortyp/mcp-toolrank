#!/usr/bin/env bash
set -euo pipefail
python_bin="${PYTHON:-.venv/bin/python}"
selected=$("$python_bin" -m toolrank.calibrate)
for spec in 'bge-small-en-v1.5 BAAI/bge-small-en-v1.5' 'toolscout-bge-small nuhmanpk/toolscout-bge-small' 'Tool-Embed-0.6B Lux1997/Tool-Embed-0.6B' "candidate artifacts/$selected"; do
    read -r name model <<< "$spec"
    options=()
    if [[ "$name" == "Tool-Embed-0.6B" ]]; then
        options=(--batch 16 --max-length 512)
    fi
    "$python_bin" -m toolrank.benchmark --model "$model" "${options[@]}" \
        --output "results/$name-toolret.json" --rows ".work/$name-toolret-rows.json"
done
