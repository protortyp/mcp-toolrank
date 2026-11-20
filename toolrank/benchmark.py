import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from toolrank.core import REX, load_model, metrics, read_json, write_json
from toolrank.etom import load as etom_corpus


def toolret():
    root = Path(".work/toolret")
    docs = {}
    queries = []
    for path in sorted(root.glob("*/tools-*.parquet")):
        for row in pq.read_table(path).to_pylist():
            key = str(row["id"])
            if key in docs:
                raise ValueError(f"duplicate tool ID {key}")
            docs[key] = row["documentation"]
    for path in sorted(root.glob("*/queries-*.parquet")):
        for row in pq.read_table(path).to_pylist():
            labels = row["labels"]
            if isinstance(labels, str):
                labels = json.loads(labels)
            gold = {str(label["id"]): int(label["relevance"]) for label in labels}
            queries.append((str(row["id"]), row["query"], path.parent.name, gold))
    if len(docs) != 44453 or len(queries) != 7961 or len({family for _, _, family, _ in queries}) != 35:
        raise ValueError(f"incomplete ToolRet download: {len(docs)} tools, {len(queries)} queries")
    return docs, queries


def local_corpus(split, schema):
    catalog = read_json(Path("data") / f"{split}.json")
    docs = {}
    queries = []
    for family in catalog:
        for tool in family["tools"]:
            tool_id = f'{family["family"]}/{tool["name"]}'
            name, desc = tool["name"], tool["description"]
            docs[tool_id] = {
                "plain": f"{name}: {desc}",
                "mcp": f'{{"name": {json.dumps(name)}, "description": {json.dumps(desc)}, "inputSchema": {{"type": "object", "properties": {{}}}}}}',
                "openai": json.dumps({"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object"}}}),
            }[schema]
            for index, query in enumerate(tool["queries"]):
                queries.append((f"{tool_id}/{index}", query, family["family"], {tool_id: 1}))
        for index, query in enumerate(family.get("none", [])):
            queries.append((f'{family["family"]}/none/{index}', query, family["family"], {}))
    return docs, queries


def mean_metrics(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def encode(model, texts, batch, query=False):
    kwargs = dict(batch_size=batch, normalize_embeddings=True, convert_to_tensor=True,
                  show_progress_bar=isinstance(texts, list) and len(texts) > 1000)
    method = model.encode_query if query else model.encode_document
    return method(texts, **kwargs).float()


def run(args):
    # CPU and CUDA benchmarks run in separate processes for isolated peak RSS.
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    model = load_model(args.model, args.device, {"dtype": getattr(torch, args.dtype)})
    if args.max_length:
        model.max_seq_length = args.max_length
    elif args.model == REX:
        model.max_seq_length = 8192

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    performance = None
    if not args.skip_profile:
        sample = read_json("data/performance.json")
        for _ in range(8):
            encode(model, sample[0], 1, query=True)
        sync()
        timings = []
        for _ in range(4):
            for query in sample:
                start = time.perf_counter()
                encode(model, query, 1, query=True)
                sync()
                timings.append(time.perf_counter() - start)
        batch_sample = sample * 8
        encode(model, batch_sample, 16, query=True)
        sync()
        start = time.perf_counter()
        encode(model, batch_sample, 16, query=True)
        sync()
        duration = time.perf_counter() - start
        performance = {
            "device": args.device,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cpu_threads": args.threads,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), None),
            "query_p50_ms": 1000 * float(np.median(timings)),
            "query_p95_ms": 1000 * float(np.quantile(timings, 0.95)),
            "queries_per_second_batch16": len(batch_sample) / duration,
            "peak_rss_after_profiling_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        }
    output = {
        "model": args.model, "parameters": sum(p.numel() for p in model.parameters()),
        "precision": str(next(model.parameters()).dtype), "split": args.split, "schema": args.schema,
        "max_seq_length": model.max_seq_length, "retrieval_batch": args.batch, "performance": performance,
    }
    if args.profile_only:
        return output

    if args.split == "etom":
        docs, queries, output["dataset_audit"] = etom_corpus(args.schema)
    elif args.split == "modern-dev":
        if not args.corpus or args.schema != "plain":
            raise ValueError("modern-dev requires --corpus and the plain schema")
        dataset = read_json(args.corpus)
        output["corpus_sha256"] = sha256(Path(args.corpus).read_bytes()).hexdigest()
        docs = {key: f"{tool['name']}: {tool['description']}" for key, tool in dataset["tools"].items()}
        queries = [(row["id"], row["query"], "+".join(row["families"]),
                    dict.fromkeys(row["positives"], 1)) for row in dataset["queries"]]
    elif args.split == "toolret":
        docs, queries = toolret()
    else:
        docs, queries = local_corpus(args.split, args.schema)
    if not docs or not queries:
        raise ValueError("empty evaluation catalog")
    ids = sorted(docs)
    missing = {key for _, _, _, gold in queries for key in gold if key not in docs}
    start = time.perf_counter()
    doc_vectors = encode(model, [docs[key] for key in ids], args.batch)
    sync()
    output["documents_per_second"] = len(docs) / (time.perf_counter() - start)
    print(f"Embedded {len(docs)} documents at {output['documents_per_second']:.1f} per second", flush=True)

    by_family = defaultdict(list)
    rows = []
    for offset in range(0, len(queries), 128):
        chunk = queries[offset:offset + 128]
        query_vectors = encode(model, [query for _, query, _, _ in chunk], args.batch, query=True)
        scores = query_vectors.to(device) @ doc_vectors.to(device).T
        # Stable ties resolve by sorted tool ID.
        top = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :10]
        values = scores.gather(1, top).tolist()
        for (query_id, _, family, gold), indices, similarities in zip(chunk, top.tolist(), values):
            ranked = [ids[index] for index in indices]
            row = {"id": query_id, "family": family, "top1": ranked[0], "score": similarities[0], "answerable": any(v > 0 for v in gold.values())}
            if row["answerable"]:
                row["metrics"] = metrics(ranked, gold)
                if args.split == "etom":
                    row["metrics"].update({f"hit@{k}": float(any(key in gold for key in ranked[:k]))
                                           for k in (1, 5, 10)})
                by_family[family].append(row["metrics"])
            rows.append(row)
    output.update(
        documents=len(docs), queries=len(queries), missing_judged_tool_ids=len(missing),
        metrics=mean_metrics([row["metrics"] for row in rows if row["answerable"]]),
        per_family={family: mean_metrics(values) for family, values in by_family.items()},
        benchmark_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        benchmark_cuda_peak_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
    )
    if args.rows:
        write_json(args.rows, rows)
    if args.threshold is not None:
        positives = [r for r in rows if r["answerable"]]
        negatives = [r for r in rows if not r["answerable"]]
        if not positives or not negatives:
            raise ValueError("abstention evaluation needs both answerable and unanswerable queries")
        output["abstention"] = {
            "threshold": args.threshold,
            "answerable_coverage": sum(r["score"] >= args.threshold for r in positives) / len(positives),
            "none_false_accept_rate": sum(r["score"] >= args.threshold for r in negatives) / len(negatives),
            "answerable_correct_at_1_with_threshold": sum(r["score"] >= args.threshold and r["metrics"].get("hit@1", r["metrics"]["recall@1"] == 1) for r in positives) / len(positives),
        }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=["toolret", "dev", "test", "etom", "modern-dev"], default="toolret")
    parser.add_argument("--corpus")
    parser.add_argument("--schema", choices=["plain", "mcp", "openai"], default="plain")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument("--profile-only", action="store_true")
    profiles.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows")
    parser.add_argument("--threshold", type=float)
    arguments = parser.parse_args()
    write_json(arguments.output, run(arguments))
