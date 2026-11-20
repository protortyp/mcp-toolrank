"""Export the smaller CPU graph from a ToolRank FP32 ONNX checkpoint."""
import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize(source: Path, output: Path) -> None:
    if source.resolve() == output.resolve():
        raise ValueError("choose a separate output path for the INT8 model")
    output.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        source, output, weight_type=QuantType.QInt8, per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm", "Gather"],
    )
    onnx.checker.check_model(str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    quantize(args.source, args.output)
