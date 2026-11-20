from pathlib import Path

from toolrank.core import read_json


MODELS = (
    ("BGE small", "bge", "bge-small-en-v1.5"),
    ("ToolScout", "scout", "toolscout-bge-small"),
    ("Tool-Embed-0.6B", "rex", "Tool-Embed-0.6B"),
    ("Candidate", "candidate", "candidate"),
)


def result(name):
    return read_json(Path("results") / f"{name}.json")


def table(header, rows):
    yield "| " + " | ".join(header) + " |"
    yield "| " + " | ".join("---" for _ in header) + " |"
    for row in rows:
        yield "| " + " | ".join(map(str, row)) + " |"


def percent(value):
    return f"{100 * value:.2f}%"


def accuracy():
    for label, _, model in MODELS:
        run = result(f"{model}-toolret")
        scores = [percent(run["metrics"][key]) for key in ("recall@1", "recall@5", "recall@10", "ndcg@10")]
        yield (label, f"{run['parameters'] / 1e6:.2f}M", *scores, f"{run['documents_per_second']:.0f}")


def performance(device):
    for label, key, model in MODELS:
        stem = f"{key}-cpu" if device == "cpu" else f"{model}-toolret"
        isolated = Path("results") / f"{key}-cuda.json"
        if device == "cuda" and isolated.is_file():
            stem = isolated.stem
        bench = result(stem)["performance"]
        memory = bench["peak_rss_after_profiling_mib"] if device == "cpu" else bench["cuda_peak_allocated_mib"]
        yield (label, f"{bench['query_p50_ms']:.2f}", f"{bench['query_p95_ms']:.2f}",
               f"{bench['queries_per_second_batch16']:.0f}", f"{memory:.0f}")


def holdout():
    for label, key, _ in MODELS:
        runs = [result(f"test/{key}-{schema}") for schema in ("plain", "mcp", "openai")]
        top1 = [percent(run["metrics"]["recall@1"]) for run in runs]
        plain = runs[0]
        abstain = plain["abstention"]
        yield (label, *top1, percent(plain["metrics"]["recall@5"]), percent(plain["metrics"]["recall@10"]),
               percent(plain["metrics"]["ndcg@10"]), percent(abstain["answerable_coverage"]),
               percent(abstain["none_false_accept_rate"]))


if __name__ == "__main__":
    for title, rows, header in (
        ("ToolRet full catalog", accuracy(), ("Model", "Parameters", "R@1", "R@5", "R@10", "nDCG@10", "RTX docs/s")),
        ("CPU encoder profile", performance("cpu"), ("Model", "p50 ms", "p95 ms", "Batch-16 queries/s", "Peak RSS MiB")),
        ("RTX 3090 encoder profile", performance("cuda"), ("Model", "p50 ms", "p95 ms", "Batch-16 queries/s", "Peak CUDA MiB")),
        ("Unseen-family holdout", holdout(), ("Model", "Plain R@1", "MCP R@1", "OpenAI R@1", "Plain R@5", "Plain R@10", "Plain nDCG@10", "Plain coverage", "Plain none false accepts")),
    ):
        print(f"## {title}\n")
        print("\n".join(table(header, rows)))
        print()
