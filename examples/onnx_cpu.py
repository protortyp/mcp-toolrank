import argparse
from pathlib import Path

import onnxruntime as ort
from transformers import AutoTokenizer

from examples.local_tools import CATALOG, QUERY


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    import json
    config = json.loads((args.checkpoint / "config_sentence_transformers.json").read_text())
    max_length = json.loads((args.checkpoint / "sentence_bert_config.json").read_text())["max_seq_length"]
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.checkpoint / "onnx" / "embedding.onnx"), settings,
                                   providers=["CPUExecutionProvider"])
    texts = [config["prompts"]["query"] + QUERY]
    texts.extend(f'{tool["name"]}: {tool["description"]}' for tool in CATALOG)
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="np")
    vectors = session.run(["embedding"], {field.name: inputs[field.name] for field in session.get_inputs()})[0]
    scores = vectors[1:] @ vectors[0]
    for tool, score in zip(CATALOG, scores):
        print(tool["name"], float(score))
