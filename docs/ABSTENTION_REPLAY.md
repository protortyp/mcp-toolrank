# Tool rejection on two catalogs

**Keep cutoffs opt-in.** Each signal missed a target on both catalogs. Adding IDF coverage to cosine cut National Weather Service (NWS) false accepts from three to one at 14/24 supported coverage. The reduction was NWS-only.

Cells show supported accepts, then unsupported false accepts. NWS and library fit local development labels. API uses NWS cutoffs.

| Signal | NWS local | Library local | API from NWS |
| --- | --- | --- | --- |
| Unfiltered hybrid | 24/24 · 24/24 | 16/16 · 16/16 | 72/72 · 24/24 |
| BM25 overlap | 24/24 · 21/24 | 16/16 · 15/16 | 72/72 · 24/24 |
| Candidate cosine | 14/24 · 3/24 | 2/16 · 0/16 | 64/72 · 7/24 |
| Hybrid maximum | 12/24 · 0/24 | 2/16 · 0/16 | 49/72 · 3/24 |
| Query coverage, BM25 or hybrid | 4/24 · 1/24 | 4/16 · 1/16 | 20/72 · 1/24 |
| Cosine plus query coverage | 14/24 · 1/24 | 2/16 · 0/16 | 66/72 · 8/24 |

## Protocol

Coverage is the catalog-IDF-weighted share of distinct query terms in tool names and descriptions. Combined score maximizes cosine + 0.15 × coverage over five top hybrid matches. Candidate cosine filters matches individually. Other cutoffs gate the shortlist.

Fit fixed signals with exact binomial (95% confidence, 10% false-accept target) on 32 NWS and 36 library calibration negatives. Evaluation used separate frozen labels. Gates also required 85% supported coverage and 80% supported hit@5. Small authored NWS samples yielded Wilson 95% intervals of 4–31% (3/24) and 1–20% (1/24). Treat the difference as descriptive.

JSON records cosine margin, standardized cosine, bidirectional transfer, selective hit@5, calibration-quantile risk–coverage, data and checkpoint hashes, and runtime versions. Reported scores matched across two runs.

```sh
uv sync --project cpu --locked --extra gateway --no-editable
uv run --project cpu --no-sync python -m toolrank.abstention_replay \
  --checkpoint /path/to/toolrank-int8 --output /tmp/abstention-replay.json
```

[Protocol](../toolrank/abstention_replay_protocol.json) · [Results](../results/abstention_replay.json) · [Pinned checkpoint](https://huggingface.co/christianengel/mcp-toolrank-33m/tree/a24a121266154006e2bd2e32a41576437c3455c2)
