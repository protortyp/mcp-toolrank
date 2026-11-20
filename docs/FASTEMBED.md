# ToolRank with FastEmbed

Run the INT8 graph through FastEmbed 0.9.1 without installing PyTorch. The graph already pools and normalizes its 384-dimensional output.

```sh
uvx --from huggingface-hub==0.36.2 hf download christianengel/mcp-toolrank-33m \
  --revision 51a961aa33a2e6dfecc6845df77203778787ec4a \
  --include 'onnx/embedding-int8.onnx' 'tokenizer.json' 'tokenizer_config.json' \
  'special_tokens_map.json' 'config.json' 'config_sentence_transformers.json' \
  --local-dir artifacts/toolrank-fastembed
uv run --no-project --python 3.12 --with fastembed==0.9.1 python - <<'PY'
from pathlib import Path
import json

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

checkpoint = Path("artifacts/toolrank-fastembed")
repo = "christianengel/mcp-toolrank-33m"
TextEmbedding.add_custom_model(
    model=repo,
    pooling=PoolingType.DISABLED,
    normalization=False,
    sources=ModelSource(hf=repo),
    dim=384,
    model_file="onnx/embedding-int8.onnx",
    output_name="embedding",
    license="Apache-2.0",
    size_in_gb=0.0341,
)
model = TextEmbedding(repo, specific_model_path=str(checkpoint.resolve()), threads=4, local_files_only=True)
prefix = json.loads((checkpoint / "config_sentence_transformers.json").read_text())["prompts"]["query"]
tools = ["get_event: Find a scheduled calendar meeting.", "get_weather: Get the weather forecast."]
vectors = np.asarray(list(model.embed([prefix + "Find my appointment this afternoon", *tools])))
scores = vectors[1:] @ vectors[0]
print(tools[int(np.argmax(scores))])
PY
```

Prefix the request only. For a persistent index, embed the catalog once and use the same graph and preprocessing for queries. Rebuild the index when changing graph or tokenizer. [INT8 evaluation](ONNX_INT8.md)
