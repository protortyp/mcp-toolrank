# National Weather Service routing pilot

**No-go for automatic routing.** ToolRank hybrid cut tool JSON bytes by 92.5%. The pilot required ≥90% top-five recall, ≥80% context reduction, ≤10% false acceptance, and ≥85% supported coverage.

| System | Top-five hits / 24 | Unsupported accepted / 24 |
| --- | ---: | ---: |
| BM25 | 19 | 24 |
| BGE Small INT8 dense | 23 | 24 |
| ToolRank INT8 dense | 22 | 24 |
| BGE Small INT8 hybrid | 20 | 24 |
| ToolRank INT8 hybrid | 21 | 24 |
| BGE Small INT8 hybrid + cosine cutoff | 7 | 0 |
| ToolRank INT8 hybrid + cosine cutoff | 14 | 3 |

Order-statistic cutoffs used 32 calibration negatives, 10% target risk, and 95% confidence under exchangeability. ToolRank's hybrid-score cutoff accepted zero unsupported and 12 supported requests. The authored holdout measures retrieval. Context counts compact tool JSON only.

The live gateway discovered the full catalog. Three allowlisted NWS probes returned HTTP 200 with expected fields for station metadata, alert types, and the glossary. Other operations were discovery-only.

```sh
gzip -dc tests/nws_pilot/spec.json.gz > nws.json
PYTHONPATH=. uv run --no-sync python tests/nws_pilot/evaluate.py \
  --spec nws.json --checkpoint /path/to/toolrank-int8 --output /tmp/nws.json
PYTHONPATH=. uv run --no-sync --with './cpu[gateway]' python tests/nws_pilot/live.py \
  --spec nws.json --checkpoint /path/to/toolrank-int8 --output /tmp/nws-live.json
```

The evaluator verifies the archived OpenAPI hash. BGE used revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, the same query prefix, scoring, and calibration as ToolRank, and repository ONNX and INT8 scripts. External spend was $0.

[NWS API](https://www.weather.gov/documentation/services-web-api) · [ToolRank results](../results/nws_pilot.json) · [BGE comparison](../results/nws_bge_pilot.json) · [Live probes](../results/nws_live_pilot.json) · [Protocol](../tests/nws_pilot/protocol.json)
