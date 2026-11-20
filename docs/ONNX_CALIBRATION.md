# Calibrate no-match rejection

Run ONNX CLI `--calibrate` on separate labelled data for a fixed catalog. Each row has `query` and `relevant` tool names (empty for no match). The report counts examples clearing the cutoff by label.

```sh
uv run --locked --script examples/rank_onnx.py ./toolrank-int8 ./catalog.json \
  --calibrate ./calibration.json --catalog-precision int8 \
  --max-false-accept 0.10 --confidence 0.95
```

Apply exact `min_score` via OnnxToolRank, CLI `--min-score`, or the MCP `search_tools` call’s `min_score` argument. Keep catalog, tokenizer, graph, runtime and cache precision fixed. The report fingerprints catalog and graph inputs. `--hybrid` reranks after cosine eligibility and preserves maximum cosine.

## Risk and sample size

The binomial order statistic controls deployment false acceptance at the requested confidence given a fixed scorer and independent calibration no-matches drawn from the deployment no-match distribution. Choose risk and confidence before scores. Evaluate answerable coverage on separate labelled deployment requests. Recalibrate after catalog/scorer changes. [Method](https://pmc.ncbi.nlm.nih.gov/articles/PMC5804623/)

10% risk at 95% confidence requires 29 no-matches. Smaller samples return `insufficient_no_match_cases` and null `min_score` before loading weights.

## Reproduction

The fictional library fixture calibrated on 36 no-matches. Separate testing retained only 2/16 answerable requests and accepted 0/16 no-matches. [Fixture](../data/onnx_calibration_demo.json) · [Results](../results/onnx-calibration.json) · [ONNX setup](ONNX_INT8.md) · [MCP stdio](MCP_SERVER.md)
