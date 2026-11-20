#!/usr/bin/env bash
set -euo pipefail
python_bin="${PYTHON:-.venv/bin/python}"
selected=$("$python_bin" -m toolrank.calibrate)
for spec in 'bge BAAI/bge-small-en-v1.5' 'scout nuhmanpk/toolscout-bge-small' 'rex Lux1997/Tool-Embed-0.6B' "candidate artifacts/$selected"; do
    read -r name model <<< "$spec"
    options=()
    if [[ "$name" == rex ]]; then options=(--max-length 512); fi
    "$python_bin" -m toolrank.benchmark --model "$model" --profile-only "${options[@]}" \
        --output "results/$name-cuda.json"
done
