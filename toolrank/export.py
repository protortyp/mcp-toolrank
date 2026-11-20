import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.nn import functional as F

from toolrank.core import load_model, read_json, write_json


class EmbeddingGraph(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        features = dict(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return F.normalize(self.model(features)["sentence_embedding"], dim=1)


def export(checkpoint, record):
    torch.set_num_threads(4)
    model = load_model(str(checkpoint)).eval()
    graph = EmbeddingGraph(model)
    path = checkpoint / "onnx" / "embedding.onnx"
    path.parent.mkdir(exist_ok=True)
    names = ["input_ids", "attention_mask", "token_type_ids"]
    texts = read_json("data/performance.json")
    sample = model.tokenize(texts[:2])
    axes = {name: {0: "batch", 1: "sequence"} for name in names}
    with torch.inference_mode():
        torch.onnx.export(
            graph, tuple(sample[name] for name in names), str(path),
            input_names=names, output_names=["embedding"],
            dynamic_axes={**axes, "embedding": {0: "batch"}}, opset_version=17, dynamo=False,
        )
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    errors = []
    for raw in (texts[2:3], texts[:4], [texts[-1] * 15]):
        for prefix in ("", model.prompts["query"]):
            inputs = model.tokenize([prefix + text for text in raw])
            actual = session.run(None, {name: inputs[name].numpy() for name in names})[0]
            expected = model.encode(raw, prompt=prefix, normalize_embeddings=True)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-4)
            errors.append(float(np.max(np.abs(actual - expected))))
    with path.open("rb") as artifact:
        digest = hashlib.file_digest(artifact, "sha256").hexdigest()
    write_json(record, {"path": str(path), "bytes": path.stat().st_size, "sha256": digest,
                        "opset": 17, "onnxruntime": ort.__version__, "max_abs_error": max(errors),
                        "cases": len(errors)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--record", default="results/onnx.json")
    args = parser.parse_args()
    export(args.checkpoint, args.record)
