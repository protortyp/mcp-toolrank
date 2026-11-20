# Agent workflows across 93 tools

Seven simulated MCP stdio servers handled 12 requests. Ten required two calls, including two that used returned IDs. Success required exact IDs and arguments with gateway receipts, or abstention on either unsupported request. Final answers and public services were outside scope.

Each cell gives completed supported workflows and schema tokens saved versus all tools.

| Exposure | 1.5B JSON | 7B JSON | 7B native |
| --- | ---: | ---: | ---: |
| All tools | 0/20 · 0% | 1/10 · 0% | 0/10 · 0% |
| BM25 top ten | 2/20 · 95.1% | 1/10 · 94.8% | 1/10 · 94.7% |
| BGE Small INT8 hybrid top ten | 4/20 · 92.7% | 1/10 · 92.9% | 1/10 · 92.7% |
| ToolRank INT8 hybrid top ten | 2/20 · 94.1% | 1/10 · 93.8% | 1/10 · 94.0% |
| ToolRank hybrid, 4,096-byte cap | 0/20 · 96.5% | 1/10 · 97.3% | 1/10 · 97.2% |

The 1.5B model rejected 0/4 unsupported trials. Each 7B JSON shortlist rejected 1/2 versus 0/2 with all tools. The 1.5B shortlists passed the relative gate against zero baseline completions. Savings include early failures. Low completion warrants deferring automatic routing and retraining.

Native 7B abstained on both unsupported requests across all methods. The adapter rejected batches. At least one batch contained both correct calls despite the single-call instruction.

Q4_K_M inference ran on RTX 3090. CPU ONNX retrieval used frozen TF-IDF/BM25 weights. Shortlist timing includes both comparators. JSON used llama.cpp `ff52ee964`. Native used `7e960f40f` after the earlier build crashed with 93 tools. Temperature was zero, with seeds 11 and 23 for 1.5B and 11 for 7B. Spend was $0.

```sh
uv sync --project cpu --locked --extra gateway --no-editable
PYTHONPATH=. uv run --project cpu --no-sync python tests/mixed_catalog/agent.py \
  --bge /path/to/bge-int8 --checkpoint /path/to/toolrank-int8 \
  --gguf /path/to/qwen-1.5b.gguf --port 18747 --output /tmp/agent-1.5b.json
PYTHONPATH=. uv run --project cpu --no-sync python tests/mixed_catalog/native_agent.py \
  --bge /path/to/bge-int8 --checkpoint /path/to/toolrank-int8 \
  --gguf /path/to/qwen-7b-shards --port 18751 --output /tmp/agent-native.json
```

For 7B JSON, use `--interface json --port 18749`. Serve the GGUF with `--ctx-size 32768 --parallel 1 --no-cache-prompt --cache-ram 0`, plus `--jinja` for native.

[Qwen 1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/91cad51170dc346986eccefdc2dd33a9da36ead9) · [Qwen 7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/tree/bb5d59e06d9551d752d08b292a50eb208b07ab1f) · [Frozen cases](../tests/mixed_catalog/agent_cases.json) · [1.5B JSON results](../results/mixed_catalog_agent.json) · [7B JSON results](../results/mixed_catalog_agent_7b.json) · [7B native results](../results/mixed_catalog_native_agent.json)

## Native batches

Six fresh requests allowed independent calls in one turn. The harness checks tool IDs and exact arguments before dispatch. All tools completed 4/5 workflows and proposed a call for the unsupported request. BM25, BGE hybrid, and ToolRank hybrid each completed 3/5, abstained on the unsupported request, and saved 92.7%, 94.3%, and 95.1% of schema tokens, respectively. Each shortlist lost one exact match for adding the NWS default `units=us`. Every method failed the workflow that depended on a returned DOI.

Repeat the native command with `--interface batch`. [Protocol](../tests/mixed_catalog/batch_protocol.json) · [Results](../results/mixed_catalog_batch_agent.json)

Seed 11 reproduced outcomes, turns, calls, and token counts in all 24 trials. [Repeat](../results/mixed_catalog_batch_agent_repeat.json)
