# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.5", "onnxruntime==1.24.1", "tokenizers==0.22.2"]
# ///
"""Rank an MCP catalog with CPU ONNX Runtime."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class OnnxToolRank:
    def __init__(self, checkpoint, tools, *, threads=4, model_file="embedding-int8.onnx"):
        self.tools = sorted(tools, key=lambda item: item["name"])
        if len({item["name"] for item in self.tools}) != len(self.tools):
            raise ValueError("catalog tool names must be unique")
        self.tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
        cfg = json.loads((checkpoint / "sentence_bert_config.json").read_text())
        self.tokenizer.enable_truncation(max_length=cfg["max_seq_length"])
        pad_id = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
        prompts = json.loads((checkpoint / "config_sentence_transformers.json").read_text())
        self.prefix = prompts["prompts"]["query"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(checkpoint / "onnx" / model_file), sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.inputs = [item.name for item in self.session.get_inputs()]
        descriptions = [f"{item['name']}: {item.get('description') or ''}" for item in self.tools]
        self.docs = (np.concatenate([self.encode(descriptions[i:i + 32])
                                     for i in range(0, len(descriptions), 32)])
                     if descriptions else np.empty((0, 384), dtype=np.float32))

    def encode(self, texts):
        rows = self.tokenizer.encode_batch(texts)
        tokens = {
            "input_ids": [row.ids for row in rows],
            "attention_mask": [row.attention_mask for row in rows],
            "token_type_ids": [row.type_ids for row in rows],
        }
        return self.session.run(["embedding"],
                                {name: np.array(tokens[name], dtype=np.int64) for name in self.inputs})[0]

    def rank(self, request, *, limit=5, min_score=None):
        if limit < 0:
            raise ValueError("limit must be nonnegative")
        if not request.strip() or not self.tools or not limit:
            return []
        scores = self.docs @ self.encode([self.prefix + request])[0]
        return [{"tool": self.tools[index], "score": float(scores[index])}
                for index in np.argsort(-scores, kind="stable")[:limit]
                if min_score is None or scores[index] >= min_score]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("catalog", type=Path, help="an MCP tools/list result or an array of tool definitions")
    parser.add_argument("request")
    parser.add_argument("--model-file", default="embedding-int8.onnx")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--min-score", type=float)
    args = parser.parse_args()
    data = json.loads(args.catalog.read_text())
    tools = data["tools"] if isinstance(data, dict) else data
    retriever = OnnxToolRank(args.checkpoint, tools, model_file=args.model_file)
    print(json.dumps(retriever.rank(args.request, limit=args.limit, min_score=args.min_score), indent=2))
