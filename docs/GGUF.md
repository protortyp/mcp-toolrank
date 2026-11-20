# Run ToolRank GGUF with llama.cpp

The 36.7 MB Q8_0 export uses released 33M weights. CPU llama.cpp serves normalized 384-dimensional embeddings with CLS pooling.

```sh
uvx --from huggingface-hub==0.36.2 hf download christianengel/mcp-toolrank-33m \
  mcp-toolrank-33m-Q8_0.gguf --local-dir artifacts/toolrank-gguf
llama-server -m artifacts/toolrank-gguf/mcp-toolrank-33m-Q8_0.gguf \
  --host 127.0.0.1 --port 8080 --embedding --pooling cls --no-cache-prompt \
  -ngl 0 -c 2048 -ub 2048 -b 2048 -np 4 -t 4 -tb 4
# Second terminal

curl -fsS http://127.0.0.1:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input":"Represent this sentence for searching relevant passages: Find museum exhibits"}'
```

Prefix queries as shown. Embed unprefixed `name: description` documents. Rank by dot product. Batch text over HTTP. Pretruncate each input to ≤512 tokens including special tokens. Send one token-ID vector per request.

Held-out nDCG@10 (Q8_0/SentenceTransformers) was 89.98%/90.13% on compact ETOM and 91.97%/92.15% on independent API. [Timing and repeats](../results/gguf.json) · [Tested llama.cpp revision](https://github.com/ggml-org/llama.cpp/tree/ff52ee964d9051e07da0e89a66567f0b893129c6)

Replay with the full local SentenceTransformer snapshot and [ETOM mirror](MODERN_MCP.md).

```sh
uv sync --locked --extra cpu --group research
uv run --no-sync python -m examples.benchmark_gguf \
  artifacts/mcp-toolrank-33m http://127.0.0.1:8080 artifacts/gguf-replay.json
```
