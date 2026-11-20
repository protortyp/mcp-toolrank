# Use ToolRank with standard ONNX backends

`onnx/model.onnx` (FP32) and `onnx/model_quantized.onnx` (INT8) return `last_hidden_state` in batch, sequence, and hidden dimensions.

## Sentence Transformers

The ONNX backend defaults to FP32 and applies stored CLS pooling and normalization. Select INT8 by filename.

```sh
uv run --locked --extra cpu --group onnx-backend python
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "christianengel/mcp-toolrank-33m", backend="onnx", device="cpu",
    model_kwargs={"file_name": "onnx/model_quantized.onnx", "export": False},
)
query = model.encode_query("Find tomorrow's meetings")
tools = model.encode_document(["search_calendar: Find meetings", "get_weather: Show forecast"])
print(tools @ query)
```

## Transformers.js

With `@huggingface/transformers@4.3.2`, use `q8` for INT8 or `fp32` for FP32. Apply the query prefix shown below and format tools as `name: description`.

```js
import { pipeline } from "@huggingface/transformers";

const embed = await pipeline("feature-extraction", "christianengel/mcp-toolrank-33m", { dtype: "q8" });
const options = { pooling: "cls", normalize: true };
const query = await embed("Represent this sentence for searching relevant passages: Find tomorrow's meetings", options);
const tools = await embed(["search_calendar: Find meetings", "get_weather: Show forecast"], options);
```

## Verify the export

On single, batched, and long CPU inputs, maximum absolute embedding errors against the pooled graphs were 8.94e-8 (FP32) and 1.19e-7 (INT8). Both exports matched their pooled counterparts' rankings across 96 API queries. Dynamic INT8 scores can vary with batch composition.

```sh
uv run --locked --extra cpu --group onnx python -m toolrank.export_onnx_backbone \
  artifacts/toolrank/onnx/embedding.onnx artifacts/toolrank/onnx/model.onnx
```

[Verification results](../results/onnx-backends.json) · [Model checkpoint](https://huggingface.co/christianengel/mcp-toolrank-33m)
