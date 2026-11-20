"""Train the matched label-only and distillation arms on the audited train split."""
import argparse
from collections import defaultdict
from hashlib import file_digest
from pathlib import Path
import subprocess
from time import perf_counter

import torch
from torch.nn import functional as F

from toolrank.core import load_model, read_json, write_json
from toolrank.source_probe import normalized_name
from toolrank.train import forward
from toolrank.train_modern import multi_positive_loss

PROTOCOL = Path(__file__).with_name("micro_distillation_protocol.json")


def digest(path):
    with Path(path).open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def keys_and_judgments(dataset):
    tools = dataset["tools"]
    keys = sorted(tools)
    names = {key: normalized_name(tools[key]["name"]) for key in keys}
    groups = defaultdict(set)
    aliases = defaultdict(set)
    for key in keys:
        groups[tools[key]["repository"]].add(key)
        aliases[names[key]].add(key)
    positive, allowed = [], []
    for row in dataset["queries"]:
        relevant = set(row["positives"])
        if not relevant or not relevant <= tools.keys():
            raise ValueError("Every training row needs available positives")
        uncertain = set()
        for key in relevant:
            uncertain |= groups[tools[key]["repository"]] | aliases[names[key]]
        positive.append([key in relevant for key in keys])
        allowed.append([key in relevant or key not in uncertain for key in keys])
    return keys, torch.tensor(positive, dtype=torch.bool), torch.tensor(allowed, dtype=torch.bool)


def softened_kl(student, teacher, allowed, temperature):
    # A finite mask avoids inf * 0 in the KL calculation.
    student = (student / temperature).masked_fill(~allowed, -1e4)
    teacher = (teacher / temperature).masked_fill(~allowed, -1e4)
    return F.kl_div(student.log_softmax(dim=-1), teacher.softmax(dim=-1), reduction="batchmean")


def teacher_scores(root, questions, documents, device, length):
    with torch.inference_mode():
        teacher = load_model(str(root.resolve()), device)
        teacher.max_seq_length = length
        teacher.eval()
        queries = teacher.encode_query(questions, batch_size=64, normalize_embeddings=True, convert_to_tensor=True)
        tools = teacher.encode_document(documents, batch_size=64, normalize_embeddings=True, convert_to_tensor=True)
        return (queries.float() @ tools.float().T).cpu()


def train_variant(args, config, seed, mode, inputs):
    questions, documents, positive, allowed, scores, hard = inputs
    torch.manual_seed(seed)
    model = load_model(str(args.student.resolve()), args.device)
    model.max_seq_length = config["max_sequence_length"]
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    generator = torch.Generator().manual_seed(seed)
    alpha = config["distillation_weights"][mode]
    query_prefix = model.prompts.get("query", "")
    autocast = model.device.type == "cuda" and torch.cuda.is_bf16_supported()
    losses = []
    started = perf_counter()
    if autocast:
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(config["epochs"]):
        order = torch.randperm(len(questions), generator=generator)
        totals = dict.fromkeys(("loss", "labels", "distillation"), 0.0)
        for rows in order.split(config["batch_size"]):
            candidates = positive[rows].any(dim=0)
            candidates[hard[rows]] = True
            indices = candidates.nonzero().flatten()
            actual = positive[rows][:, indices].to(model.device)
            eligible = allowed[rows][:, indices].to(model.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=autocast):
                query_vectors = forward(model, [query_prefix + questions[i] for i in rows.tolist()])
                tool_vectors = forward(model, [documents[i] for i in indices.tolist()])
            similarity = query_vectors @ tool_vectors.T
            labels = multi_positive_loss(similarity / config["label_temperature"], actual, eligible)
            distillation = softened_kl(similarity, scores[rows][:, indices].to(model.device), eligible,
                                       config["teacher_temperature"]) if alpha else labels.new_zeros(())
            loss = (1 - alpha) * labels + alpha * distillation
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            for name, value in (("loss", loss), ("labels", labels), ("distillation", distillation)):
                totals[name] += value.detach().item() * len(rows)
        losses.append({name: value / len(questions) for name, value in totals.items()})
        print(f"{mode} seed={seed} epoch={epoch + 1} loss={losses[-1]['loss']:.5f}", flush=True)
    duration = perf_counter() - started
    directory = args.output / f"{mode}-{seed}"
    model.save_pretrained(directory)
    return {"weights_sha256": digest(directory / "model.safetensors"), "losses": losses,
            "wall_seconds": duration, "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "autocast": "bfloat16" if autocast else "float32",
            "gpu_peak_mib": torch.cuda.max_memory_allocated() / 1024**2 if autocast else None}


def run(args):
    protocol = read_json(PROTOCOL)
    for key, path in (("student", args.student), ("teacher", args.teacher)):
        if digest(path / "model.safetensors") != protocol[key]["weights_sha256"]:
            raise ValueError(f"Unexpected {key} checkpoint")
    if digest(args.train) != protocol["train_sha256"]:
        raise ValueError("Expected the audited training split")
    torch.set_num_threads(4)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(.25)
    config = protocol["training"]
    dataset = read_json(args.train)
    keys, positive, allowed = keys_and_judgments(dataset)
    tools = dataset["tools"]
    documents = [f"{tools[key]['name']}: {tools[key]['description']}" for key in keys]
    questions = [row["query"] for row in dataset["queries"]]
    started = perf_counter()
    scores = teacher_scores(args.teacher, questions, documents, args.device, config["max_sequence_length"])
    if args.device == "cuda":
        torch.cuda.empty_cache()
    negatives = allowed & ~positive
    if not negatives.any(dim=1).all():
        raise ValueError("Each row requires at least one eligible hard negative")
    hard = scores.masked_fill(~negatives, float("-inf")).argmax(dim=1)
    result = {"protocol": protocol, "protocol_sha256": digest(PROTOCOL), "source_sha256": digest(__file__),
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "examples": len(questions), "tools": len(keys), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
              "teacher_and_mining_seconds": perf_counter() - started,
              "hard_negative_catalog_coverage": len(hard.unique()), "models": {}, "external_compute_usd": 0}
    inputs = (questions, documents, positive, allowed, scores, hard)
    for seed in protocol["seeds"]:
        for mode in config["distillation_weights"]:
            result["models"][f"{mode}-{seed}"] = train_variant(args, config, seed, mode, inputs)
            write_json(args.record, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("student", "teacher", "train", "output", "record"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run(parser.parse_args())
