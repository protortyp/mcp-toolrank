# Calibrate a live gateway

For hybrid search, label requests with `query` and `relevant`, an array of eligible `server/name` IDs. Use `[]` for unsupported requests. Set the false-accept target and confidence before scoring. Fit and evaluate on separate labels.

```sh
toolrank-gateway ./gateway.json --calibrate ./calibration.json > ./cutoff.json
# Add "calibration_report": "cutoff.json" to gateway.json.
toolrank-gateway ./gateway.json --evaluate ./holdout.json
```

Use `calibration_report` in place of `min_cosine`, with a config-relative path. For `invocable_only: true` searches, add `--invocable-only` to both labeled commands. Startup rejects a scorer mismatch. Search returns `calibration_scope_mismatch` if the flag or eligible catalog differs. Fitting requires ready upstreams and at least 29 negatives for 10% risk at 95% confidence.

Check held-out coverage, false acceptance, and selective hit rate (relevant hits among accepted searches). The binomial bound assumes a fixed scorer and independent, representative negatives. [Statistical method](https://pmc.ncbi.nlm.nih.gov/articles/PMC5804623/)

## NWS holdout

The installed wheel discovered 69 tools and fitted on 32 negatives. Both commands use discovery only. On holdout, false accepts fell from 24/24 to 3/24, coverage from 24/24 to 14/24, and top-five hits from 21/24 to 14/24. The cutoff missed both targets (≤10% false acceptance, ≥85% coverage). Scorer and catalog changes invalidated the report.

```sh
TOOLRANK_OFFLINE=1 PYTHONPATH=. uv run --project cpu --no-sync python \
  tests/nws_pilot/gateway_calibration_smoke.py --gateway /path/to/toolrank-gateway \
  --checkpoint /path/to/toolrank-int8 --output /tmp/gateway-calibration.json
```

[Raw fit and holdout results](../results/gateway_calibration_smoke.json) · [NWS pilot](NWS_PILOT.md)

BM25 overlap accepted most unsupported requests. Across three catalogs, cosine, margin, and lexical coverage each missed a preset target. [Rejection replay](ABSTENTION_REPLAY.md)
