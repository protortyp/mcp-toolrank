"""Count projected tool-schema tokens with the pinned local llama.cpp tokenizer."""
import argparse
import csv
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from toolrank_cpu.gateway import limit_tool_bytes

from run import CASES, Retriever, catalog, digest


def count_tokens(port, matches):
    # Match the local agent harness, which forwards schemas and drops scores and revisions.
    projected = [{"id": item["id"], "invocable": item["invocable"], "description": item["description"],
                  "parameters": item["inputSchema"]} for item in matches]
    content = json.dumps(projected, ensure_ascii=False, separators=(",", ":"))
    body = json.dumps({"content": content, "add_special": False}).encode()
    request = Request(f"http://127.0.0.1:{port}/tokenize", data=body,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=15) as response:
        return len(json.load(response)["tokens"])


def run(checkpoint, port):
    entries = catalog()
    engine = Retriever(entries, {"toolrank": checkpoint})
    full = engine.payload([(i, None, None) for i in range(len(entries))])
    counts = {"all": [count_tokens(port, full)], "joint_top10": [], "joint_top10_4096_bytes": []}
    with CASES.open(newline="", encoding="utf-8") as source:
        for case in csv.DictReader(source, delimiter="\t"):
            ranking = engine.rank(case["intent_a"] + " " + case["intent_b"])["toolrank"]
            matches = engine.payload(ranking)
            counts["joint_top10"].append(count_tokens(port, matches))
            counts["joint_top10_4096_bytes"].append(count_tokens(port, limit_tool_bytes(matches, 4096)))
    return {key: {"sample_count": len(values), "median_tokens": float(np.median(values)),
                  "max_tokens": max(values)} for key, values in counts.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18746)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = {"tokenizer": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
               "revision": "91cad51170dc346986eccefdc2dd33a9da36ead9",
               "runtime": "llama.cpp ff52ee964", "gguf_sha256": digest(args.gguf),
               "case_sha256": digest(CASES), "source_sha256": digest(Path(__file__)),
               "scope": "Only projected tool schemas. No system prompt, query, generation or invocation.",
               "counts": run(args.checkpoint, args.port)}
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
