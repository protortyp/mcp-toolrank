# 17M training data audit

**Review query-specific labels before retraining.** The 17M and 33M hybrids shared 242 top-one misses on family-disjoint development. Only 24/252 student misses put unjudged tools from relevant repositories first. Blanket same-repository negatives risk suppressing valid alternatives.

| Hybrid, 872 requests | Judged at 1 | Any judged at 5 | Unjudged same repo at 1 |
| --- | ---: | ---: | ---: |
| 17M, seed 11 | 620 | 858 | 24 |
| 33M ToolRank | 620 | 866 | 22 |
| BGE Small | 611 | 856 | 28 |
| Untuned BGE Micro | 598 | 861 | 30 |

The teacher recovered 11 of 14 student top-five misses. The student recovered three teacher misses. Paired difference (student minus teacher) was −0.92 points (95% family bootstrap interval −1.55 to 0.00, nine families). Incomplete judgments leave feasibility open.

Adding all 263 training distractors expanded development to 297 tools with fixed requests and positives. The 17M and 33M had 584 vs 556 judged hits at one and 837 vs 849 at five. Paired differences were +3.21 points (95% interval −0.33 to +5.69) and −1.38 (−2.11 to 0.00). Distractor relevance was unjudged.

## Training-only review

On 500/5,098 training rows, 33M ranked an unjudged same-repository tool above all judged positives. Queue held 3,407 ranked candidates across 550 distinct ordered pairs. The 12-pair review prioritized frequency and teacher score, finding three clear negatives, three plausible alternatives, and six ambiguous. Counts describe this sample. [Review and pinned definitions](../results/micro_training_audit_review.json)

Reviewed distinctions included hosted image URLs vs base64, timezone conversion vs relative time, and listing models vs generating images. Review labels stayed local and outside training.

## Reproduce

Use audited splits and original checkpoints. Optional `--review-queue` writes an identifier-only training queue. Two RTX 3090 runs reproduced metrics and queue order.

```sh
uv sync --locked --extra cu128 --group research
HF_HUB_OFFLINE=1 uv run --no-sync python -m toolrank.audit_micro_errors \
  --train /path/to/train.json --dev /path/to/dev.json \
  --labels /path/to/archived-models/labels-11 --teacher /path/to/toolrank-fp32 \
  --bge /path/to/bge-small --micro /path/to/bge-micro \
  --output /tmp/micro-training-audit.json
```

[Counts, hashes, and paired results](../results/micro_training_audit.json) · [Data provenance](MODERN_MCP.md) · [17M pilot](MICRO_DISTILLATION.md)

## Family-weighted training

**No-go.** On 297 development tools, weighting raised mean family-macro Hit@5 by 0.043 points (gate +1.00) and nDCG@10 by 0.096 (gate +0.50). Compact nDCG@10 fell 0.057 points, within the 0.50-point limit.

| Seed | 297-tool Hit@5 change, points | 297-tool nDCG@10 change, points |
| --- | ---: | ---: |
| 11 | +0.035 | +0.068 |
| 23 | +0.087 | +0.181 |
| 37 | +0.006 | +0.039 |

Both arms shared 17M initialization per seed, 5,098 requests across 36 families, row and batch order, masks, hard negatives, optimizer, serialization, and one epoch. Inverse-square-root family frequency weighted query loss, capped at 3× the preliminary mean before renormalizing. Both catalogs shared 872 disjoint development queries and lexical weights. Expansion added 263 training tools with unjudged relevance.

Six RTX 3090 retrains reproduced the mean expanded Hit@5 difference exactly. Repeat nDCG@10 gain was 0.094 points. Both runs missed the same gates. Family-macro averages nine families equally. The largest supplies 382 queries, so query averages can change rankings. Holdouts stayed sealed. External spend was $0.

```sh
uv sync --locked --extra cu128 --group research
HF_HUB_OFFLINE=1 uv run --no-sync python -m toolrank.family_weighting \
  --student /path/to/bge-micro --teacher /path/to/toolrank-fp32 \
  --train /path/to/train.json --dev /path/to/dev.json \
  --output /tmp/toolrank-family-models --record /tmp/toolrank-family-weighting.json
```

[Protocol](../toolrank/family_weighting_protocol.json) · [Results](../results/family_weighting.json) · [Repeat](../results/family_weighting_replay.json)
