"""Count potential MCP supervision without saving queries, tool calls, or traces."""
from collections import Counter, defaultdict
import json
import re
from urllib.parse import urlparse

import pyarrow.parquet as pq

from toolrank.core import read_json, write_json


ATLAS_ROOT = ".work/next/atlas-servers.json"
SHARD = ".work/next/toucan.parquet"
PERMISSIVE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "0bsd", "unlicense"}


def normalized_name(name):
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def github_id(url):
    parsed = urlparse(url.strip())
    if parsed.hostname != "github.com":
        return None
    pieces = parsed.path.strip("/").removesuffix(".git").split("/")
    if len(pieces) != 2 or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", piece) for piece in pieces):
        return None
    return "/".join(pieces).casefold()


def resolve(label, names):
    if label in names:
        return label
    candidates = [name for name in names if name.endswith("-" + label)]
    return candidates[0] if len(candidates) == 1 else None


def run():
    atlas = read_json(ATLAS_ROOT)["mcpServers"]
    atlas_names = {normalized_name(name) for name in atlas}
    counts = Counter()
    source_rows = Counter()
    source_tools = set()
    source_licenses = Counter()
    shard = pq.ParquetFile(SHARD)
    fields = ["question", "available_tools", "target_tools", "metadata"]
    for batch in shard.iter_batches(batch_size=512, columns=fields):
        for row in batch.to_pylist():
            counts["rows"] += 1
            names = {item["function"]["name"] for item in json.loads(row["available_tools"])}
            labels = [label.strip() for label in row["target_tools"].split(",") if label.strip()]
            if not labels or not row["question"].strip():
                continue
            targets = {resolve(label, names) for label in labels}
            if None in targets:
                continue
            counts["all_labels_resolved"] += 1
            counts["single_tool_resolved"] += int(len(targets) == 1)
            servers = json.loads(row["metadata"])["mcp_servers"]
            source = defaultdict(set)
            for server in servers:
                info = server["server_info"]
                repo = github_id(info.get("repository_url") or "")
                license_name = (info.get("license") or "").casefold().strip()
                raw_names = server["remote_server_response"].get("tool_names") or []
                server_name = server["server_name"]
                for tool in raw_names:
                    key = normalized_name(f"{server_name}-{tool}")
                    source[key].add((repo, license_name, normalized_name(server_name)))
            sources = {target: source[normalized_name(target)] for target in targets}
            if any(len(origins) != 1 for origins in sources.values()):
                continue
            counts["all_labels_linked_to_server"] += 1
            target_sources = {target: next(iter(options)) for target, options in sources.items()}
            origins = set(target_sources.values())
            if any(name in atlas_names for _, _, name in origins):
                counts["direct_atlas_server_name_overlap"] += 1
                continue
            if all(repo and license in PERMISSIVE for repo, license, _ in origins):
                counts["all_targets_have_candidate_permissive_sources"] += 1
                for repo, license, _ in origins:
                    source_rows[repo] += 1
                    source_licenses[license] += 1
                source_tools.update((target_sources[target][0], target) for target in targets)
    write_json(".work/next/source-inventory.json", dict(source_rows.most_common()))
    write_json("results/mcp-source-probe.json", {
        "shard_revision": "0df3cf37f2abefb380370cfb02eabea2a35ae782",
        "shard": "Kimi-K2/train-00000-of-00040.parquet",
        "atlas_template_revision": "f8c0e6e8adf0b76a70cc706ad0bb9f96252728eb",
        "atlas_template_server_names": len(atlas_names),
        "counts": dict(counts),
        "candidate_distinct_repositories": len(source_rows),
        "candidate_distinct_target_tools": len(source_tools),
        "candidate_license_declarations": dict(source_licenses),
        "top_candidate_repositories": dict(source_rows.most_common(15)),
        "verification_scope": "Atlas exclusion uses direct server-name matching only. Repository aliases and source license revisions still need verification.",
    })


if __name__ == "__main__":
    run()
