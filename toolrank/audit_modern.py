"""Measure normalized overlaps across training, development, and frozen ETOM inputs."""
import argparse
import json
from pathlib import Path
import re

from toolrank.core import read_json, write_json
from toolrank.etom import FILES, ROOT, load, reserved_families
from toolrank.verify_sources import provider_alias


def normalized(text):
    return re.sub(r"\W+", " ", text.casefold()).strip()


def inventories(dataset):
    tools = dataset["tools"].values()
    return {
        "repository": {tool["repository"] for tool in tools},
        "tool_text": {normalized(f"{tool['name']}: {tool['description']}") for tool in tools},
        "query": {normalized(row["query"]) for row in dataset["queries"]},
    }


def exact_holdout_overlap(training, dataset, root):
    from toolrank.replay_hybrid import corpus

    documents, queries = corpus(dataset, root)
    overlap = 0
    for document in documents.values():
        representations = {normalized(document)}
        try:
            parsed = json.loads(document)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            definition = parsed.get("function", parsed)
            if isinstance(definition, dict) and definition.get("name") and definition.get("description"):
                representations.add(normalized(f"{definition['name']}: {definition['description']}"))
        overlap += bool(representations & training["tool_text"])
    return {f"train_{dataset}_tool_text": overlap,
            f"train_{dataset}_query": sum(normalized(text) in training["query"] for _, text, _, _ in queries)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(".work/next/modern-expanded"))
    parser.add_argument("--etom", type=Path, default=ROOT)
    parser.add_argument("--additional-holdouts", action="store_true", help="check exact API and ToolRet overlap")
    parser.add_argument("--output", default="results/mcp-split-independence.json")
    args = parser.parse_args()
    train = inventories(read_json(args.dataset / "train.json"))
    dev = inventories(read_json(args.dataset / "dev.json"))
    docs, queries, _ = load("plain", args.etom)
    reserved = reserved_families(read_json(args.etom / FILES[0]))
    etom = {"repository": set(reserved["repository_slugs"]),
            "tool_text": {normalized(text) for text in docs.values()},
            "query": {normalized(text) for _, text, _, _ in queries}}
    aliases = {provider_alias(name) for name in reserved["names"]}
    report = {f"{left}_{right}_{kind}": len(a[kind] & b[kind])
              for left, right, a, b in (("train", "dev", train, dev),
                                        ("train", "etom", train, etom), ("dev", "etom", dev, etom))
              for kind in ("repository", "tool_text", "query")}
    for split, refs in (("train", train), ("dev", dev)):
        report[f"{split}_etom_repository_alias"] = sum(provider_alias(repo.split("/")[1]) in aliases
                                                       for repo in refs["repository"])
    if args.additional_holdouts:
        for dataset in ("api", "toolret"):
            report.update(exact_holdout_overlap(train, dataset, args.etom))
    write_json(args.output, report)
    if any(report.values()):
        raise ValueError("detected an exact split overlap")
