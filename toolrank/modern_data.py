"""Freeze a family-disjoint development split from staged, source-checked rows."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from sklearn.feature_extraction.text import HashingVectorizer

from toolrank.core import read_json, write_json
from toolrank.source_probe import normalized_name


def digest(value):
    return sha256(value.encode()).hexdigest()


def prepare(rows, output):
    parent = {}
    owners = {}
    def canonical(family):
        prior = parent.setdefault(family, family)
        if prior != family:
            parent[family] = canonical(prior)
        return parent[family]

    for row in rows:
        for tool in row["positives"]:
            repo = tool["repository"]
            basename = normalized_name(repo.split("/")[1])
            for key in (("basename", basename), ("description", normalized_name(tool["name"]), tool["description"])):
                owner = owners.setdefault(key, repo)
                left, right = sorted((canonical(repo), canonical(owner)))
                parent[right] = left

    groups = sorted({canonical(repo) for repo in parent}, key=lambda repo: digest("mcp-development-v1/" + repo))
    if len(groups) < 5:
        raise ValueError("need at least five independent repository groups")
    development = set(groups[:max(2, len(groups) // 5)])
    split_rows = defaultdict(list)
    discarded = 0
    for row in rows:
        families = {canonical(tool["repository"]) for tool in row["positives"]}
        memberships = {family in development for family in families}
        if len(memberships) != 1:
            discarded += 1
            continue
        split = "dev" if True in memberships else "train"
        split_rows[split].append((row, tuple(sorted(families))))

    vectorizer = HashingVectorizer(analyzer="char", ngram_range=(4, 4), n_features=2**19, alternate_sign=False)
    training_text = [row["query"].casefold() for row, _ in split_rows["train"]]
    dev_text = [row["query"].casefold() for row, _ in split_rows["dev"]]
    similarity = vectorizer.transform(dev_text) @ vectorizer.transform(training_text).T
    keep = similarity.max(axis=1).toarray().ravel() < 0.9
    split_rows["dev"] = [item for item, keep_item in zip(split_rows["dev"], keep) if keep_item]
    report = {"near_duplicate_dev_queries_removed": int((~keep).sum()), "repository_groups": len(groups), "development_groups": sorted(development),
              "cross_split_queries_dropped": discarded, "splits": {}}
    for split in ("train", "dev"):
        tools = {}
        queries = []
        for row, families in split_rows[split]:
            relevant = []
            for tool in row["positives"]:
                key = f"{tool['repository']}::{tool['name']}::{digest(tool['description'])[:16]}"
                tools[key] = tool
                relevant.append(key)
            queries.append({"id": row["id"], "query": row["query"], "families": list(families),
                            "positives": sorted(set(relevant))})
        if not queries or not tools:
            raise ValueError(f"empty split: {split}")
        path = output / f"{split}.json"
        write_json(path, {"tools": tools, "queries": queries})
        report["splits"][split] = {"queries": len(queries), "tools": len(tools),
                                   "sha256": sha256(path.read_bytes()).hexdigest(),
                                   "query_counts_by_family": dict(Counter(
                                       family for item in queries for family in item["families"]))}
    write_json(output / "split-report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = prepare(read_json(args.rows), args.output)
    print({key: value["queries"] for key, value in summary["splits"].items()})
