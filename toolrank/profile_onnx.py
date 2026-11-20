"""Measure warmed batch-one latency of both published CPU ONNX graphs."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.rank_onnx import OnnxToolRank
from toolrank.core import read_json, write_json


def profile(checkpoint: Path, threads: int):
    runtimes = {
        name: OnnxToolRank(checkpoint, [], threads=threads, model_file=filename)
        for name, filename in (("fp32", "embedding.onnx"), ("int8", "embedding-int8.onnx"))
    }
    short = read_json("data/performance.json")
    long = [query + " Include the owner, date, and status for every matching record." * 30 for query in short]
    result = {}
    for workload, queries in (("short", short), ("long", long)):
        times = {key: [] for key in runtimes}
        for runtime in runtimes.values():
            for query in queries:
                runtime.encode([runtime.prefix + query])
        for i, query in enumerate(queries * 8):
            order = ("fp32", "int8") if i % 2 else ("int8", "fp32")
            for key in order:
                runtime = runtimes[key]
                start = perf_counter()
                runtime.encode([runtime.prefix + query])
                times[key].append(1000 * (perf_counter() - start))
        result[workload] = {key: {"p50_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}
                            for key, values in times.items()}
    return {"cpu_threads": threads, "batch_size": 1, "samples_per_workload": len(short) * 8, "latency": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, profile(args.checkpoint, args.threads))
