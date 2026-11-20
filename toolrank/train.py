import argparse
import hashlib
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from toolrank.core import BGE, PREFIX, load_model, read_json, write_json


def training_examples(path):
    docs = {}
    families = {}
    rows = []
    for family in read_json(path):
        name = family["family"]
        families[name] = []
        for tool in family["tools"]:
            key = f'{name}/{tool["name"]}'
            docs[key] = f'{tool["name"]}: {tool["description"]}'
            families[name].append(key)
            rows.extend((query, key, name) for query in tool["queries"])
    return docs, families, rows


def forward(model, texts):
    inputs = model.tokenize(texts)
    inputs = {name: value.to(model.device) for name, value in inputs.items()}
    return F.normalize(model(inputs)["sentence_embedding"].float())


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    model = load_model(BGE, args.device)
    model.max_seq_length = 128
    docs, families, rows = training_examples("data/train.json")
    ids = sorted(docs)

    # Static mining uses the initialization and training catalog alone.
    model.eval()
    if args.negatives == "mined":
        queries = model.encode_query([q for q, _, _ in rows], normalize_embeddings=True)
        vectors = model.encode_document([docs[key] for key in ids], normalize_embeddings=True)
        similarities = queries @ vectors.T
        order = np.argsort(-similarities, axis=1, kind="stable")
        negatives = [
            [ids[i] for i in indices if ids[i] != positive][:args.hard_count]
            for indices, (_, positive, _) in zip(order, rows)
        ]
    elif args.negatives == "family":
        negatives = [[key for key in families[family] if key != positive] for _, positive, family in rows]
    else:
        negatives = [[] for _ in rows]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    epoch_losses = []
    rng = random.Random(args.seed)
    for _ in range(args.epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), args.batch):
            batch = order[start:start + args.batch]
            positives = {rows[i][1] for i in batch}
            tool_ids = sorted(positives.union(*(negatives[i] for i in batch)))
            columns = {key: index for index, key in enumerate(tool_ids)}
            labels = torch.tensor([columns[rows[i][1]] for i in batch], device=model.device)
            allowed = torch.tensor(
                [[key in positives or key in negatives[i] for key in tool_ids] for i in batch],
                device=model.device,
            )
            optimizer.zero_grad(set_to_none=True)
            query_vectors = forward(model, [PREFIX + rows[i][0] for i in batch])
            tool_vectors = forward(model, [docs[key] for key in tool_ids])
            logits = (query_vectors @ tool_vectors.T) / args.temperature
            loss = F.cross_entropy(logits.masked_fill(~allowed, float("-inf")), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        epoch_losses.append(total / len(rows))

    output = Path(args.output)
    model.save_pretrained(str(output))
    write_json(args.record, {
        "base": BGE,
        "base_revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "training_sha256": hashlib.sha256(Path("data/train.json").read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if model.device.type == "cuda" else None,
        "optimizer": "AdamW",
        "max_seq_length": model.max_seq_length,
        "config": vars(args),
        "examples": len(rows),
        "tools": len(docs),
        "epoch_loss": epoch_losses,
        "weights_sha256": hashlib.file_digest((output / "model.safetensors").open("rb"), "sha256").hexdigest(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--negatives", choices=["inbatch", "family", "mined"], default="family")
    parser.add_argument("--hard-count", type=int, default=3)
    train(parser.parse_args())
