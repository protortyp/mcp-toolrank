import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from toolrank.export_onnx_backbone import extract


class BackboneExportTests(unittest.TestCase):
    def test_extract_keeps_tokens_and_removes_first_token_pooling(self):
        graph = helper.make_graph(
            [
                helper.make_node("Identity", ["input"], ["tokens"]),
                helper.make_node("Gather", ["tokens", "position"], ["embedding"],
                                 name="/model/model.1/Gather", axis=1),
            ],
            "pooled",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", "sequence", 2])],
            [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 2])],
            [helper.make_tensor("position", TensorProto.INT64, [], [0])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
        with tempfile.TemporaryDirectory() as work:
            source, target = Path(work) / "pooled.onnx", Path(work) / "backbone.onnx"
            onnx.save(model, source)
            extract(source, target)
            session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
            assert session.get_outputs()[0].name == "last_hidden_state"
            tokens = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
            np.testing.assert_array_equal(session.run(None, {"input": tokens})[0], tokens)


if __name__ == "__main__":
    unittest.main()
