"""Load the pinned ETOM release for local static retrieval evaluation."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlparse

from toolrank.core import read_json, write_json


REVISION = "790f28727f567b88918420d163c524d46644a694"
ROOT = Path(".work/next/etom")
FILES = ("mcp_registry.json", "level_1.json", "level_2.json", "level_5.json")
LEVELS = (1, 2, 5)


def reserved_families(registry):
    from toolrank.source_probe import normalized_name

    names = set()
    repos = set()
    for server in registry["servers"]:
        names.add(normalized_name(server["server_name"]))
        parsed = urlparse(server["server_url"])
        parts = parsed.path.strip("/").split("/")
        if parsed.hostname == "glama.ai" and len(parts) == 4 and parts[:2] == ["mcp", "servers"] and parts[2].startswith("@"):
            repos.add(f"{parts[2][1:]}/{parts[3]}".casefold())
            names.add(normalized_name(parts[3]))
    return {"names": sorted(names - {""}), "repository_slugs": sorted(repos)}


def load(schema, root=ROOT):
    registry = read_json(root / FILES[0])
    docs = {}
    for server in registry["servers"]:
        family = server["server_name"]
        for tool in server["tools"]:
            name = tool["tool_name"]
            description = tool["tool_description"]
            tool_id = f"{family}::{name}"
            if tool_id in docs:
                raise ValueError(f"duplicate ETOM tool ID {tool_id}")
            if schema == "plain":
                text = f"{name}: {description}"
            elif schema == "mcp":
                text = json.dumps({"name": name, "description": description, "inputSchema": tool["input_schema"]})
            elif schema == "openai":
                text = json.dumps({"type": "function", "function": {
                    "name": name, "description": description, "parameters": tool["input_schema"],
                }})
            else:
                raise ValueError(f"unknown tool schema: {schema}")
            docs[tool_id] = text

    if len(docs) != 2375 or len(registry["servers"]) != 491:
        raise ValueError("unexpected ETOM registry size")
    queries = []
    removed = Counter()
    seen_ids = set()
    for level in LEVELS:
        for row in read_json(root / f"level_{level}.json"):
            query_id = f"l{level}/{row['query_id']}"
            if query_id in seen_ids:
                raise ValueError(f"duplicate ETOM query ID {query_id}")
            seen_ids.add(query_id)
            entries = row["ground_truth_tools"]
            if level == 2:
                # The release provides overlapping equivalence groups for the same intent.
                entries = [entry for group in entries for entry in group]
            gold = {entry["tool_id"]: 1 for entry in entries}
            if bool(gold) != (level != 5):
                raise ValueError(f"unexpected ETOM judgment type for {query_id}")
            if gold.keys() - docs.keys():
                removed[f"l{level}"] += 1
                continue
            queries.append((query_id, row["query"], f"etom-l{level}", gold))
    if len(seen_ids) != 1645:
        raise ValueError("unexpected ETOM query count")
    audit = {
        "revision": REVISION,
        "sha256": {name: sha256((root / name).read_bytes()).hexdigest() for name in FILES},
        "documents": len(docs),
        "queries_by_level": dict(Counter(q[2] for q in queries)),
        "skipped_queries_with_incomplete_qrels": dict(removed),
    }
    return docs, queries, audit


if __name__ == "__main__":
    _, _, audit = load("plain")
    write_json("results/etom-intake.json", audit)
    write_json(ROOT / "reserved-families.json", reserved_families(read_json(ROOT / FILES[0])))
