"""Measure normalized overlaps across training, development, and frozen ETOM inputs."""
import argparse
from pathlib import Path
import re

from toolrank.core import read_json, write_json
from toolrank.etom import ROOT, load
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(".work/next/modern-expanded"))
    args = parser.parse_args()
    train = inventories(read_json(args.dataset / "train.json"))
    dev = inventories(read_json(args.dataset / "dev.json"))
    docs, queries, _ = load("plain")
    reserved = read_json(ROOT / "reserved-families.json")
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
    write_json("results/mcp-split-independence.json", report)
    if any(report.values()):
        raise ValueError("detected an exact split overlap")
