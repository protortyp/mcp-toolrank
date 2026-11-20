#!/usr/bin/env bash
set -euo pipefail
python_bin="${PYTHON:-.venv/bin/python}"
"$python_bin" -m toolrank.audit_modern
candidate=$("$python_bin" -m toolrank.select_modern)
for schema in mcp plain openai; do
    for spec in 'bge BAAI/bge-small-en-v1.5' 'scout nuhmanpk/toolscout-bge-small' 'rex Lux1997/Tool-Embed-0.6B' "candidate $candidate"; do
        read -r name model <<< "$spec"
        calibration=".work/next/etom/$name-$schema-calibration.json"
        "$python_bin" -m toolrank.benchmark --split dev --schema "$schema" --model "$model" \
            --max-length 512 --skip-profile --output ".work/next/etom/$name-$schema-calibration-metrics.json" \
            --rows "$calibration"
        threshold=$("$python_bin" -m toolrank.calibrate --threshold "$calibration")
        "$python_bin" -m toolrank.benchmark --split etom --schema "$schema" --model "$model" \
            --max-length 512 --batch 24 --skip-profile --threshold "$threshold" \
            --output "results/etom/$name-$schema.json" --rows ".work/next/etom/$name-$schema-rows.json"
    done
    "$python_bin" -m toolrank.etom_compare --schema "$schema"
done
