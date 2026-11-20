"""Train a 512-token dual encoder from a frozen, audited local training split."""
import argparse
from hashlib import file_digest, sha256
from pathlib import Path

import torch

from toolrank.core import BGE, MICRO, load_model, read_json, write_json
from toolrank.fetch import REVISIONS
from toolrank.source_probe import normalized_name
from toolrank.train import forward


def multi_positive_loss(logits, positive, allowed):
    masked = logits.masked_fill(~allowed, float("-inf"))
    log_denominator = masked.logsumexp(dim=1)
    positive_mean = logits.masked_fill(~positive, 0).sum(dim=1) / positive.sum(dim=1)
    return (log_denominator - positive_mean).mean()


def run(args):
    dataset = read_json(args.training)
    tools = dataset["tools"]
    rows = dataset["queries"]
    if not rows or not all(row["positives"] for row in rows):
        raise ValueError("each training query needs a positive")
    names = {key: normalized_name(tool["name"]) for key, tool in tools.items()}
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    model = load_model(args.base, args.device)
    prefix = model.prompts.setdefault("query", "")
    model.max_seq_length = 512
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    shuffle = torch.Generator().manual_seed(args.seed)
    epoch_losses = []
    use_bfloat16 = model.device.type == "cuda" and torch.cuda.is_bf16_supported()
    for _ in range(args.epochs):
        order = torch.randperm(len(rows), generator=shuffle).tolist()
        accumulated = 0.0
        for start in range(0, len(rows), args.batch):
            batch = [rows[index] for index in order[start:start + args.batch]]
            keys = sorted({key for row in batch for key in row["positives"]})
            positive = []
            allowed = []
            for row in batch:
                relevant = set(row["positives"])
                aliases = {names[key] for key in relevant}
                providers = {tools[key]["repository"] for key in relevant}
                positive.append([key in relevant for key in keys])
                allowed.append([
                    key in relevant or (names[key] not in aliases and tools[key]["repository"] not in providers)
                    for key in keys
                ])
            positive = torch.tensor(positive, dtype=torch.bool, device=model.device)
            allowed = torch.tensor(allowed, dtype=torch.bool, device=model.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                query_vectors = forward(model, [prefix + row["query"] for row in batch])
                tool_vectors = forward(model, [f"{tools[key]['name']}: {tools[key]['description']}" for key in keys])
            logits = (query_vectors @ tool_vectors.T) / args.temperature
            loss = multi_positive_loss(logits, positive, allowed)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            accumulated += loss.detach().item() * len(batch)
        epoch_losses.append(accumulated / len(rows))
        print(f"Epoch {len(epoch_losses)} loss {epoch_losses[-1]:.5f}", flush=True)
    model.save_pretrained(args.output)
    with (Path(args.output) / "model.safetensors").open("rb") as stream:
        weights = file_digest(stream, "sha256").hexdigest()
    write_json(args.record, {
        "base": args.base, "base_revision": REVISIONS[f"models/{args.base}"],
        "training_sha256": sha256(Path(args.training).read_bytes()).hexdigest(),
        "config": vars(args), "training_autocast": "bfloat16" if use_bfloat16 else "float32",
        "epoch_losses": epoch_losses, "weights_sha256": weights,
        "queries": len(rows), "tools": len(tools), "max_seq_length": 512,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", required=True)
    parser.add_argument("--base", choices=[BGE, MICRO], default=BGE)
    parser.add_argument("--output", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    run(parser.parse_args())
