# Pilot reproduction

From the repository root with `uv`, curl, and an RTX 3090 or compatible CUDA device on Linux x86-64:

```sh
uv sync --locked --extra cu128 --group research --group onnx
uv run --locked --no-sync python -m toolrank.fetch
uv run --locked --no-sync python -m toolrank.audit
uv run --locked --no-sync bash toolrank/pilot.sh
uv run --locked --no-sync bash toolrank/evaluate.sh
uv run --locked --no-sync bash toolrank/toolret.sh
uv run --locked --no-sync bash toolrank/profile_cuda.sh
uv run --locked --no-sync python -m toolrank.compare
uv run --locked --no-sync python -m toolrank.diagnose
uv run --locked --no-sync python -m toolrank.report
```

The lockfile resolves Python 3.12 and separate PyTorch builds. Use the `cpu` extra for CPU inference. The original pilot's dependency snapshot remains in `requirements-cu128.txt`. The lockfile preserves the core model runtime versions and can resolve newer transitive dependencies.

Downloads are public and require no token. The mirror pins model and dataset revisions. It normalizes ToolScout’s module metadata for the pinned SentenceTransformers runtime. The full ToolRet catalog and local model checkpoints live under ignored directories. Public benchmark rows with upstream IDs also stay local. Local evaluation rows reference the project’s original test data.
