"""Match Toucan descriptions against pinned upstream source without running servers."""
import argparse
from collections import Counter, defaultdict
from functools import cache
from hashlib import sha1, sha256
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

import pyarrow.parquet as pq

from toolrank.core import read_json, write_json
from toolrank.etom import ROOT as ETOM_ROOT
from toolrank.source_probe import ATLAS_ROOT, SHARD, github_id, normalized_name, resolve


ARCHIVES = Path(".work/next/source-archives")
PROVENANCE = Path("results/mcp-upstream-license-spotcheck.json")
TEXT_EXTENSIONS = {".md", ".py", ".ts", ".js", ".go", ".json", ".yaml", ".yml", ".rs", ".cs"}


def provider_alias(name):
    return normalized_name(name).replace("mcp", "").replace("server", "")


def compact(text):
    return " ".join(text.split())


def git_blob(data):
    return sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def archive_path(repo, revision):
    return ARCHIVES / f"{repo.replace('/', '--')}--{revision}.tar.gz"


@cache
def source_files(repo, revision, license_path, license_blob):
    archive = archive_path(repo, revision)
    if not archive.is_file():
        return {}
    files = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not PurePosixPath(member.name).parts[0].endswith(f"-{revision}"):
                raise ValueError(f"archive revision mismatch for {repo}")
            if not member.isfile() or member.size > 512 * 1024:
                continue
            path = PurePosixPath(*PurePosixPath(member.name).parts[1:])
            if path.as_posix() != license_path and (path.suffix not in TEXT_EXTENSIONS or
                                                   any(part in {"vendor", "dist", "node_modules"} for part in path.parts)):
                continue
            with tar.extractfile(member) as handle:
                raw = handle.read()
            files[path.as_posix()] = (git_blob(raw), compact(raw.decode("utf-8", errors="replace")))
    notice = files.get(license_path)
    if notice is None or notice[0] != license_blob:
        return {}
    return files


@cache
def verify(repo, revision, license_path, license_blob, name, description):
    expected = compact(description)
    if len(expected) < 25:
        return None
    name_pattern = re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])")
    for path, (blob, content) in sorted(source_files(repo, revision, license_path, license_blob).items()):
        if path != license_path and expected in content and name_pattern.search(content):
            return {"repository": repo, "commit": revision, "path": path, "source_blob": blob,
                    "license": "MIT", "license_path": license_path, "license_blob": license_blob}
    return None


def run(shards, report, output_rows, provenance):
    deny = read_json(ETOM_ROOT / "reserved-families.json")
    reserved_names = set(deny["names"])
    reserved_names.update(normalized_name(name) for name in read_json(ATLAS_ROOT)["mcpServers"])
    reserved_repos = set(deny["repository_slugs"])
    reserved_aliases = {provider_alias(name) for name in reserved_names} - {""}
    manifest = read_json(provenance)
    counts = Counter()
    matches = {}
    eligible_rows = {}
    conflicted = set()
    fields = ["available_tools", "target_tools", "metadata", "uuid", "question"]
    batches = ((shard, batch) for shard in shards for batch in
               pq.ParquetFile(shard).iter_batches(batch_size=512, columns=fields))
    for shard, batch in batches:
        for row in batch.to_pylist():
            counts["rows"] += 1
            specs = {item["function"]["name"]: item["function"] for item in json.loads(row["available_tools"])}
            labels = [name.strip() for name in row["target_tools"].split(",") if name.strip()]
            targets = {resolve(name, specs) for name in labels}
            if not targets or None in targets:
                continue
            source = defaultdict(set)
            for server in json.loads(row["metadata"])["mcp_servers"]:
                family = normalized_name(server["server_name"])
                repo = github_id(server["server_info"].get("repository_url") or "")
                for name in server["remote_server_response"].get("tool_names") or []:
                    key = normalized_name(f"{server['server_name']}-{name}")
                    source[key].add((repo, family, name))
            positives = []
            for target in sorted(targets):
                options = source[normalized_name(target)]
                if len(options) != 1:
                    continue
                repo, family, name = next(iter(options))
                if not repo:
                    continue
                if (repo in reserved_repos or provider_alias(family) in reserved_aliases
                        or provider_alias(repo.split("/")[1]) in reserved_aliases):
                    counts["reserved_target_mentions"] += 1
                    continue
                record = manifest.get(repo, {})
                if record.get("spdx") != "MIT" or "observed_default_head" not in record:
                    continue
                counts["eligible_target_mentions"] += 1
                description = specs[target].get("description")
                if not isinstance(description, str):
                    counts["target_mentions_without_text_descriptions"] += 1
                    continue
                identity = (repo, name, sha256(compact(description).encode()).hexdigest())
                evidence = verify(repo, record["observed_default_head"], record["path"], record["blob"], name, description)
                if evidence:
                    matches[identity] = evidence
                    positives.append({"repository": repo, "name": name, "description": compact(description),
                                      "source_path": evidence["path"], "source_commit": evidence["commit"]})
            counts["rows_with_a_verified_target"] += int(bool(positives))
            if len(positives) != len(targets):
                continue
            counts["rows_with_all_targets_verified"] += 1
            if not output_rows:
                continue
            question = compact(row["question"] or "")
            if not question or re.search(r"(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?:https?://|/home/|/Users/|[A-Za-z]:\\)|"
                                         r"\b[A-Za-z0-9_-]{32,}\b)", question):
                counts["candidate_rows_rejected_by_text_filter"] += 1
                continue
            key = question.casefold()
            label = tuple(sorted((tool["repository"], tool["name"], tool["description"]) for tool in positives))
            previous = eligible_rows.get(key)
            if previous and previous[0] != label:
                conflicted.add(key)
            elif not previous:
                eligible_rows[key] = (label, {"id": row["uuid"], "shard": Path(shard).name, "query": question,
                                              "positives": positives})
    if output_rows:
        data = [item[1] for key, item in sorted(eligible_rows.items()) if key not in conflicted]
        counts["candidate_unique_requests"] = len(data)
        counts["candidate_conflicting_requests"] = len(conflicted)
        write_json(output_rows, data)
    write_json(report, {
        "shards": [Path(shard).name for shard in shards],
        "counts": dict(counts),
        "matches": [{"tool_name": name, "description_sha256": digest, **evidence}
                    for (_, name, digest), evidence in sorted(matches.items())],
        "scope": "Exact whitespace-normalized descriptions and tool names in the same pinned file, with a verified MIT license blob. This checks source evidence and direct identity exclusions.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", action="append", type=Path)
    parser.add_argument("--output", default="results/mcp-source-verification.json")
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--provenance", type=Path, default=PROVENANCE)
    args = parser.parse_args()
    run(args.shard or [Path(SHARD)], args.output, args.rows, args.provenance)
