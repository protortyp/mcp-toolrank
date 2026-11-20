# Gateway preview

The CPU source archive rebuilds the identical wheel without model downloads. With cached dependencies and an empty model cache, isolated Python 3.12 and 3.13 demos passed a clock receipt between two local stdio servers. On Linux, the desktop bundle’s stdio manifest launched and matched the expected tool. [Release checks](../results/gateway_release_smoke.json)

Run from anywhere.

```sh
uvx --python 3.12 --from '/path/to/toolrank_cpu-0.2.0-py3-none-any.whl[gateway]' \
  toolrank-gateway --demo
```

Local `--checkpoint` requires model files and notices. The bundle verifies the model and includes all 43 license and attribution files from the pinned Hub tree.

```sh
uv build cpu --out-dir dist/staged
python3 bundle/build.py dist/staged/mcp-toolrank-0.7.1.mcpb \
  --checkpoint /path/to/pinned-model-snapshot
```

Chromium checked model-free BM25, no-overlap behavior, imported schemas, and all three API holdout modes with pinned local artifacts. [Browser verification](../results/browser_release_smoke.json)

The preview serves one trusted local client. Hybrid retrieved more accurately on ETOM. The NWS cutoff missed its targets. [Gateway](GATEWAY.md) · [Calibration](GATEWAY_CALIBRATION.md) · [Technical article](RELEASE_ARTICLE.md)
