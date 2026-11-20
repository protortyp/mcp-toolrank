"""Supplement the frozen CPU screen with a matched FP32 RTX 3090 ToolRet run."""
import argparse
from pathlib import Path
import subprocess

import torch

from toolrank.baseline_comparison import compare
from toolrank.core import load_model, read_json, write_json
from toolrank.train_micro_distillation import PROTOCOL, digest


def run(args):
    protocol = read_json(PROTOCOL)
    record = read_json(args.training_record)
    if record["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Training protocol mismatch")
    checkpoints = {"toolrank": (args.teacher, protocol["teacher"]["weights_sha256"]),
                   "micro": (args.candidate, record["models"]["labels-11"]["weights_sha256"])}
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.25)
    models = {}
    for name, (path, expected) in checkpoints.items():
        if digest(path / "model.safetensors") != expected:
            raise ValueError(f"Unexpected model: {name}")
        model = load_model(str(path.resolve()), "cuda", {"dtype": torch.float32})
        model.max_seq_length = 512
        models[name] = model
    scores = compare("toolret", models, "cuda", 32, paired_models=("toolrank", "micro"))
    return {"gpu": torch.cuda.get_device_name(), "dtype": "float32", "batch_size": 32,
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": digest(__file__), "training_record_sha256": digest(args.training_record),
            "checkpoints": {name: value[1] for name, value in checkpoints.items()},
            "torch": torch.__version__, "supplemental_to_cpu_gate": True,
            "scores": scores, "external_compute_usd": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher", "candidate", "training-record", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, run(args))
