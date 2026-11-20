"""Screen INT8 MatMulNBits activation computation on development inputs only."""
import argparse
from pathlib import Path
import platform
from subprocess import check_output
from tempfile import TemporaryDirectory

import onnx

from examples.rank_onnx import OnnxToolRank
from toolrank.core import write_json
from toolrank.onnx_batch_pilot import DEV, FILES, dev_probe, digest

SCREENS = {"max_embedding_delta": 0.0001, "max_p95_multiple": 1.15}


def change_accuracy(source, target):
    model = onnx.load(source)
    nodes = [node for node in model.graph.node if node.op_type == "MatMulNBits"
             and node.domain == "com.microsoft"]
    if not nodes:
        raise ValueError("Expected a ToolRank MatMulNBits graph")
    for node in nodes:
        accuracy, = [attr for attr in node.attribute if attr.name == "accuracy_level"]
        if accuracy.i != 1:
            raise ValueError("The source must use FP32 activation computation")
        accuracy.i = 4
    onnx.checker.check_model(model)
    onnx.save(model, target)
    return len(nodes)


def run(args):
    from statistics import median

    with TemporaryDirectory(dir=args.candidates) as temporary:
        root = Path(temporary)
        model_name = "embedding-weight-int4-level4.onnx"
        for filename in ("tokenizer.json", "sentence_bert_config.json", "config_sentence_transformers.json"):
            (root / filename).symlink_to((args.candidates / filename).resolve())
        (root / "onnx").mkdir()
        source = args.candidates / "onnx" / FILES["weight-int4"]
        target = root / "onnx" / model_name
        nodes = change_accuracy(source, target)
        options = {"dynamic-int8": (args.int8, FILES["dynamic-int8"]),
                   "weight-int4-fp32-compute": (args.candidates, FILES["weight-int4"]),
                   "weight-int4-int8-compute": (root, model_name)}
        results = {name: [] for name in options}
        # Rotate graph order across rounds to reduce host warmup and load effects.
        pairs = list(options.items())
        for offset in range(len(pairs)):
            for name, (path, filename) in pairs[offset:] + pairs[:offset]:
                encoder = OnnxToolRank(path, [], threads=4, model_file=filename)
                results[name].append(dev_probe(encoder))
        median_p95 = {name: median(row["p95_query_ms"] for row in runs) for name, runs in results.items()}
        trial = results["weight-int4-int8-compute"]
        passed = all(row["max_embedding_delta"] <= SCREENS["max_embedding_delta"] for row in trial)
        ratio = median_p95["weight-int4-int8-compute"] / median_p95["dynamic-int8"]
        return {"development_only": True, "platform": platform.platform(), "development_sha256": digest(DEV),
                "source_graph_sha256": digest(source), "trial_graph_sha256": digest(target),
                "trial_bytes": target.stat().st_size, "changed_nodes": nodes, "screens": SCREENS,
                "rounds": results, "median_p95_ms": median_p95, "trial_p95_multiple": ratio,
                "passes_screen": passed and ratio <= SCREENS["max_p95_multiple"],
                "source_sha256": digest(__file__),
                "revision": check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "external_compute_usd": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, run(args))
