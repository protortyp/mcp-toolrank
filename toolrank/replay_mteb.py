"""Evaluate released ToolRank and BGE on the pinned MTEB ToolRetrieval task."""

import argparse
from pathlib import Path

import mteb
from mteb.models.cache_wrappers import CachedEmbeddingWrapper
import torch

MODELS = {
    "toolrank": ("christianengel/mcp-toolrank-33m", "ec27cb99405cf5076aba6090843f40f4aa51eb96"),
    "bge-small": ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"),
}
DATASET_REVISION = "76d45e560059754e289ea202462865a585679619"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model_id, revision = MODELS[args.model]
    task = mteb.get_task("ToolRetrieval")
    if task.metadata.dataset["revision"] != DATASET_REVISION:
        raise ValueError("MTEB ToolRetrieval dataset revision differs from the replay")
    model = mteb.get_model(model_id, revision=revision, device=args.device)
    print(task.metadata.name, model_id, revision, flush=True)
    cached = CachedEmbeddingWrapper(model, args.output.parent / "embeddings" / args.model)
    result = mteb.evaluate(
        cached, task, encode_kwargs={"batch_size": 64}, cache=None,
        show_progress_bar=False,
    )
    if len(result.task_results) != 1 or result.exceptions:
        raise RuntimeError("ToolRetrieval did not produce one successful result")
    scores = result.task_results[0]
    subsets = {row["hf_subset"] for row in scores.scores["test"]}
    if subsets != set(task.hf_subsets):
        raise RuntimeError("ToolRetrieval results are missing benchmark subsets")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_disk(args.output)
    print(scores.get_score(), flush=True)


if __name__ == "__main__":
    main()
