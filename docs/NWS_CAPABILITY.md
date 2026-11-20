# NWS capability verification

**Keep the verifier out of the default gateway.** It met the 10% false-accept limit and missed the 85% correct-shortlist and 25 ms CPU p95 targets.

| System | Required tool in top five / 30 | Unsupported accepted / 32 |
| --- | ---: | ---: |
| Hybrid without rejection | 27 | 32 |
| Hybrid with cosine cutoff | 22 | 3 |
| Hybrid with Qwen3 0.6B verifier | 23 | 1 |

The shortlist missed three required tools. Verification rejected four more. Scoring used GPU BF16. CPU verification alone took 2,192 ms p95 across 13 warmed calibration requests, with five pairs per batch and four threads.

Calibration alone set inputs, candidate count, and cutoff rule before evaluation on 62 fresh authored requests from the same catalog. The evaluation made no tool calls. MiniLM-L2 stayed in calibration. At 12.3 ms p95, it retained 18/24 correct shortlists against a 20/24 target and accepted 3/32 unsupported requests.

```sh
gzip -dc tests/nws_pilot/spec.json.gz > /tmp/nws.json
PYTHONPATH=. uv run --locked --extra cu128 --group research --group onnx python \
  tests/nws_pilot/verify_capability.py --spec /tmp/nws.json \
  --checkpoint /path/to/toolrank-int8 --model /path/to/qwen3-reranker \
  --profile-cpu --output /tmp/nws-capability.json
```

Qwen3 revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`. Checksums are in the results. External compute cost was $0.

[Protocol](../tests/nws_pilot/capability_protocol.json) · [Results](../results/nws_capability.json) · [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
