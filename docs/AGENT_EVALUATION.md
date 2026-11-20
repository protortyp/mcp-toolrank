# Local agent evaluation

Five-tool shortlists cut schema tokens by 85–87% versus all tools. Each strategy ran 11 synthetic tasks twice at temperature 0 (seeds 11 and 23) against three MCP stdio processes. Success required every expected call with the correct target and arguments, or abstention when appropriate.

| Strategy | Completed | Mean tool JSON tokens | Mean prompt tokens |
| --- | ---: | ---: | ---: |
| All tools | 12/22 | 2080 | 2228 |
| BM25 | 13/22 | 302 | 450 |
| BGE Small dense | 16/22 | 301 | 450 |
| ToolRank dense | 16/22 | 301 | 450 |
| BGE Small hybrid | 14/22 | 278 | 414 |
| ToolRank hybrid | 16/22 | 302 | 451 |

Fixtures cover duplicate names, permissions, similar operations, dependent calls, catalog refresh, and unavailable tools. Each hybrid handled one of two no-tool task types. The harness records missing tools, selection and argument failures, and upstream errors. Incorrect proposals stop before dispatch.

Run [Qwen2.5 1.5B Q4_K_M](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/91cad51170dc346986eccefdc2dd33a9da36ead9) with llama.cpp `ff52ee964`.

```sh
llama-server --model /path/to/model.gguf --host 127.0.0.1 --port 18742 \
  --ctx-size 8192 --threads 6 --n-gpu-layers 0
PYTHONPATH=. uv run --locked --extra cpu --group research --with './cpu[gateway]' \
  python tests/agent_eval/run.py --toolrank /path/to/checkpoint \
  --agent-model /path/to/model.gguf --agent-revision 91cad51170dc346986eccefdc2dd33a9da36ead9 \
  --llama-revision ff52ee964 --output results/agent_evaluation.json
```

Latency includes inference, startup, and invocation. Token counts use Qwen’s tokenizer and server usage. Trials measure tool workflow completion on these fixtures. [Raw results](../results/agent_evaluation.json)
