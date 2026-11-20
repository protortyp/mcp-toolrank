# LangChain and Pydantic AI

Each turn uses the request and latest result to shortlist up to five invocable tools for the native chat API. The loop checks the first proposal against the shortlist, invokes its original tool ID and revision, and discards extras.

The read-only demo passes a Tokyo timestamp from a loopback HTTP clock to a stdio tool that computes minutes since midnight. BM25 ranks tools. The local OpenAI-compatible endpoint receives schemas and results.

```sh
uv sync --project cpu --locked --extra gateway --no-editable
uv sync --locked --extra cpu --group examples
```

From the checkout root, run the clock and Qwen in separate terminals.

```sh
uv run --project cpu --no-sync python examples/gateway_http_clock.py
llama-server --model /path/to/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --host 127.0.0.1 --port 18753 --jinja --ctx-size 8192
```

```sh
PYTHONPATH=. uv run --no-sync --with './cpu[gateway]' python -m examples.langchain_agent \
  --gateway cpu/.venv/bin/toolrank-gateway
PYTHONPATH=. uv run --no-sync --with './cpu[gateway]' python -m examples.pydantic_agent \
  --gateway cpu/.venv/bin/toolrank-gateway
```

Verified with llama.cpp `7e960f40f` and `--threads 6 --n-gpu-layers 99 --parallel 1 --no-cache-prompt --cache-ram 0`. Each framework passed two live trials on this workflow. Replace `-m examples.pydantic_agent` with `tests/gateway/framework_smoke.py` to verify call order, exact timestamp handoff, and final answers.

[Smoke results](../results/gateway_framework_smoke.json) · [Demo config](../examples/gateway_agent_config.json) · [Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/tree/bb5d59e06d9551d752d08b292a50eb208b07ab1f) · [LangChain models](https://docs.langchain.com/oss/python/langchain/models) · [Pydantic AI models](https://ai.pydantic.dev/api/models/openai/)
