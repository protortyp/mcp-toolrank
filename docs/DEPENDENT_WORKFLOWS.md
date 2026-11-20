# Dependent MCP calls

| Tool exposure and dispatch | Supported completed / 7 | Mean first-turn schema tokens |
| --- | ---: | ---: |
| All tools, reject multi-call output | 3 | 20,969 |
| All tools, consider first proposal | 6 | 20,969 |
| All tools, compact schemas | 5 | 16,307 |
| ToolRank top ten | 5 | 1,383 |
| ToolRank top ten, compact schemas | 5 | 1,160 |

Serial dispatch raised full-catalog completion from 3/7 to 6/7. It withheld six proposals and replanned after the first response. Each method abstained on 1/2 unsupported requests. Compaction saved 22.2% of first-turn schema tokens and lost one completion. The gate required ≥15% savings and at least as many completions as full-schema serial dispatch. Keep full schemas in gateway responses. ToolRank top ten saved 93.4% with 5/7 completions. Both top-ten variants included the expected tool on every attempted turn. Their failures occurred after retrieval.

Seven tasks required a second call using returned arguments. The nine requests used 93 tools across seven simulated MCP stdio servers. Scoring covered only exact allowlisted calls dispatched by the gateway and abstention. Explicit documented defaults matched omitted defaults. All arms shared a native tool call prompt. Strict dispatch rejected multi-call output. Compaction removed schema documentation, including parameter guidance, and kept constraints, defaults, tool descriptions, and candidate IDs. [JSON Schema metadata](https://json-schema.org/understanding-json-schema/reference/metadata)

Qwen2.5-7B Q4_K_M ran on RTX 3090 with llama.cpp `7e960f40f`, seed 17, temperature zero, and CPU ONNX ToolRank retrieval. Serve the pinned shards on port 18751 with `--jinja --ctx-size 32768 --n-gpu-layers 99 --parallel 1 --no-cache-prompt --cache-ram 0`. Results record input and checkpoint hashes. External spend was $0. [Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/tree/bb5d59e06d9551d752d08b292a50eb208b07ab1f)

```sh
uv sync --project cpu --locked --extra gateway --no-editable
PYTHONPATH=. uv run --project cpu --no-sync python tests/mixed_catalog/workflow.py \
  --checkpoint /path/to/toolrank-int8 --gguf /path/to/qwen-7b-shards \
  --output /tmp/dependent_workflows.json
```

[Protocol](../tests/mixed_catalog/workflow_protocol.json) · [Cases](../tests/mixed_catalog/workflow_cases.json) · [Results](../results/dependent_workflows.json)

Seed 23 reproduced full-catalog completion on the same tasks, 3/7 strict and 6/7 first proposal. Rerun with `--protocol tests/mixed_catalog/workflow_repeat_protocol.json`. [Repeat results](../results/dependent_workflows_repeat.json)
