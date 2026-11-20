"""Extract the token-level transformer from an exported ToolRank graph."""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper
from onnx.utils import Extractor


def extract(source: Path, target: Path):
    if source.resolve() == target.resolve():
        raise ValueError("choose a different output file")
    model = onnx.load(source)
    if [out.name for out in model.graph.output] != ["embedding"]:
        raise ValueError("expected the pooled ToolRank graph")
    pooling = [
        node for node in model.graph.node if node.op_type == "Gather"
        and node.name == "/model/model.1/Gather"
    ]
    if len(pooling) != 1:
        raise ValueError("expected the ToolRank first-token pooling operation")
    hidden = pooling[0].input[0]
    model = onnx.shape_inference.infer_shapes(model)
    shape = next(value for value in model.graph.value_info if value.name == hidden)
    dims = shape.type.tensor_type.shape.dim
    if len(dims) != 3 or shape.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError("expected floating-point token vectors")
    backbone = Extractor(model).extract_model([item.name for item in model.graph.input], [hidden])
    backbone.graph.node.append(helper.make_node("Identity", [hidden], ["last_hidden_state"]))
    backbone.graph.output[0].name = "last_hidden_state"
    onnx.checker.check_model(backbone)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(backbone, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    extract(args.source, args.target)
