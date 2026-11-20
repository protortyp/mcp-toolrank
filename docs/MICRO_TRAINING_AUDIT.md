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
