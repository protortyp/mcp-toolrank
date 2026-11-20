"""Export fixed-scale INT8 and weight-only INT4 graphs for the ONNX pilot."""
import argparse
from hashlib import file_digest
import json
import logging
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter

from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from tokenizers import Tokenizer

PROTOCOL = Path(__file__).with_name("onnx_batch_protocol.json")
CONFIG_FILES = ("tokenizer.json", "sentence_bert_config.json", "config_sentence_transformers.json")
FILES = {"fp32": "embedding.onnx", "dynamic-int8": "embedding-int8.onnx",
         "static-int8": "embedding-static-int8.onnx", "weight-int4": "embedding-weight-int4.onnx"}


def digest(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def examples(path, prefix):
    groups = json.loads(path.read_text())
    texts = [f"{tool['name']}: {tool['description']}" for row in groups for tool in row["tools"]]
    queries = [prefix + query for row in groups for tool in row["tools"] for query in tool["queries"]]
    return texts + queries


class Calibration(CalibrationDataReader):
    def __init__(self, tokenizer, texts):
        import numpy as np

        rows = []
        texts = sorted(texts, key=len)
        for offset in range(0, len(texts), 8):
            tokens = tokenizer.encode_batch(texts[offset:offset + 8])
            rows.append({name: np.asarray([getattr(row, field) for row in tokens], dtype=np.int64)
                         for name, field in (("input_ids", "ids"), ("attention_mask", "attention_mask"),
                                             ("token_type_ids", "type_ids"))})
        self.rows = rows
        self.rewind()

    def get_next(self):
        return next(self.current, None)

    def rewind(self):
        self.current = iter(self.rows)


def static_export(source, target, root, protocol):
    settings = protocol["candidates"]["static-int8"]
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=json.loads((root / "sentence_bert_config.json").read_text())["max_seq_length"])
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
    prefix = json.loads((root / "config_sentence_transformers.json").read_text())["prompts"]["query"]
    dataset = Path(protocol["calibration"])
    texts = examples(dataset, prefix)
    calibration = Calibration(tokenizer, texts)
    with TemporaryDirectory() as directory:
        prepared = Path(directory) / "inferred.onnx"
        quant_pre_process(source, prepared, skip_optimization=True)
        quantize_static(prepared, target, calibration, quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        per_channel=settings["per_channel"], op_types_to_quantize=settings["ops"],
                        calibrate_method=CalibrationMethod.MinMax,
                        calibration_providers=["CPUExecutionProvider"],
                        extra_options={"MatMulConstBOnly": True, "ActivationSymmetric": True,
                                       "WeightSymmetric": True, "CalibMaxIntermediateOutputs": 8})
    return {"examples": len(texts), "batches": len(calibration.rows), "calibration_sha256": digest(dataset)}


def weight_export(source, target, settings):
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    converter = MatMulNBitsQuantizer(str(source), bits=settings["bits"], block_size=settings["block_size"],
                                    is_symmetric=settings["symmetric"], accuracy_level=settings["accuracy_level"],
                                    op_types_to_quantize=tuple(settings["ops"]))
    converter.process()
    converter.model.save_model_to_file(str(target), use_external_data_format=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("static-int8", "weight-int4"), required=True)
    args = parser.parse_args()
    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)
    protocol = json.loads(PROTOCOL.read_text())
    source = args.fp32 / "onnx" / FILES["fp32"]
    published = args.int8 / "onnx" / FILES["dynamic-int8"]
    if digest(source) != protocol["source_sha256"] or digest(published) != protocol["dynamic_int8_sha256"]:
        raise ValueError("Expected the pinned ToolRank FP32 and dynamic INT8 graphs")
    target = args.output / "onnx" / FILES[args.variant]
    target.parent.mkdir(parents=True, exist_ok=True)
    # The three graphs must share identical input and prompt settings.
    for filename in CONFIG_FILES:
        if digest(args.fp32 / filename) != digest(args.int8 / filename):
            raise ValueError(f"Checkpoint configs differ for {filename}")
        if not (args.output / filename).exists():
            shutil.copyfile(args.fp32 / filename, args.output / filename)
    start = perf_counter()
    metadata = {}
    if args.variant == "static-int8":
        metadata = static_export(source, target, args.output, protocol)
    else:
        weight_export(source, target, protocol["candidates"][args.variant])
    import onnx

    onnx.checker.check_model(str(target))
    metadata |= {"wall_seconds": perf_counter() - start, "bytes": target.stat().st_size,
                 "sha256": digest(target), "protocol_sha256": digest(PROTOCOL),
                 "source_sha256": digest(Path(__file__))}
    (args.output / f"{args.variant}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
