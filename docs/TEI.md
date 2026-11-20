# Serve ToolRank with TEI

Download the pinned checkpoint. CPU TEI serves on loopback using offline loading and full-precision token ONNX (`onnx/model.onnx`). Standalone ranking uses pooled `onnx/embedding.onnx`.

```sh
uvx --from huggingface-hub==0.36.2 hf download christianengel/mcp-toolrank-33m \
  --revision fb65bd562f54c1b3e6433586bfc423592782f1fb \
  --include config.json config_sentence_transformers.json tokenizer.json tokenizer_config.json \
  special_tokens_map.json sentence_bert_config.json model.safetensors modules.json 1_Pooling/config.json onnx/model.onnx \
  --local-dir artifacts/toolrank-tei

docker run --rm -p 127.0.0.1:8080:80 -e HF_HUB_OFFLINE=1 \
  -v "$(pwd)/artifacts/toolrank-tei:/model:ro" \
  ghcr.io/huggingface/text-embeddings-inference@sha256:2538ea1c9640d3763b15af668039d24172d063b42337b0c27796fc2be180c78d \
  --model-id /model --dtype float32 --max-client-batch-size 32
```

Use `"prompt_name":"query"` for requests and unprompted `name: description` rows for tools. Rank normalized vectors by dot product.

```sh
curl -fsS http://127.0.0.1:8080/embed -H 'Content-Type: application/json' \
  -d '{"inputs":["Find museum exhibits open today"],"prompt_name":"query","normalize":true}'
curl -fsS http://127.0.0.1:8080/embed -H 'Content-Type: application/json' \
  -d '{"inputs":["list_exhibits: List museum exhibits"],"normalize":true}'
```

TEI 1.9.4 on Linux x86-64 matched SentenceTransformers’ ordered top three on nine authored queries against nine documents, including over-length inputs. Cosine difference peaked below 4.18e-7.

```sh
uv sync --locked --extra cpu
uv run --no-sync python -m examples.validate_tei http://127.0.0.1:8080 artifacts/toolrank-tei /tmp/tei.json
```

[Validation results](../results/tei.json)
