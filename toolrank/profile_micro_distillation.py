"""Measure the first registered student's INT8 CPU latency against released ToolRank."""
import argparse
from pathlib import Path
import platform
import shutil
from statistics import median
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from examples.rank_onnx import OnnxToolRank, checkpoint_checksums
from toolrank.core import read_json, write_json
from toolrank.export import export
from toolrank.quantize_onnx import quantize
from toolrank.train_micro_distillation import PROTOCOL, digest

PERFORMANCE = Path("data/performance.json")
GRAPH = "embedding-int8.onnx"


def variant(source, destination, record):
    shutil.copytree(source, destination)
    cfg = destination / "config_sentence_transformers.json"
    existing = read_json(cfg)
    existing.setdefault("prompts", {}).setdefault("query", "")
    write_json(cfg, existing)
    export(destination, record)
    quantize(destination / "onnx/embedding.onnx", destination / "onnx" / GRAPH)


def cycle(runtimes, queries):
    names = tuple(runtimes)
    samples = {name: [] for name in names}
    for runtime in runtimes.values():
        for query in queries:
            runtime.encode([runtime.prefix + query])
    for i, query in enumerate(queries * 8):
        order = names if i % 2 == 0 else names[::-1]
        for name in order:
            runtime = runtimes[name]
            start = perf_counter()
            runtime.encode([runtime.prefix + query])
            samples[name].append(1000 * (perf_counter() - start))
    return {name: {"p50_ms": median(times), "p95_ms": float(np.quantile(times, .95)), "samples": len(times)}
            for name, times in samples.items()}


def run(args):
    protocol = read_json(PROTOCOL)
    baseline = args.int8 / "onnx" / GRAPH
    expected = "ba0e544669eac502cac535d3ed53b9cf5b530b5fd929d4a7bddd787abe08ea33"
    if digest(baseline) != expected:
        raise ValueError("Unexpected published INT8 graph")
    first = f"distilled-{protocol['seeds'][0]}"
    source = args.models / first
    training = read_json(args.training_record)
    if digest(source / "model.safetensors") != training["models"][first]["weights_sha256"]:
        raise ValueError("Student checkpoint is not the registered seed")
    with TemporaryDirectory() as temporary:
        root = Path(temporary) / "student"
        metadata = Path(temporary) / "fp32.json"
        variant(source, root, metadata)
        paths = {"toolrank": args.int8, first: root}
        runtimes = {name: OnnxToolRank(path, [], threads=4) for name, path in paths.items()}
        short = read_json(PERFORMANCE)
        long = [(text + " Include the date and status for each result.") * 40 for text in short]
        profile = {name: cycle(runtimes, texts) for name, texts in (("short", short), ("long", long))}
        graphs = {name: {"bytes": (path / "onnx" / GRAPH).stat().st_size,
                         "sha256": checkpoint_checksums(path, GRAPH)} for name, path in paths.items()}
        verification = read_json(metadata)
        verification.pop("path")
    return {"development_probe": True, "student": first, "threads": 4, "workload_sha256": digest(PERFORMANCE),
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": digest(__file__), "platform": platform.platform(),
            "cpu": next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                        if line.startswith("model name")),
            "graphs": graphs, "student_fp32_parity": verification, "profile": profile, "external_compute_usd": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("int8", "models", "training-record", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, run(args))
