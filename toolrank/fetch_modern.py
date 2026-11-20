"""Fetch pinned public research inputs for the larger MCP experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

from toolrank.core import read_json
from toolrank.etom import FILES, REVISION, ROOT
from toolrank.fetch import download
from toolrank.source_probe import ATLAS_ROOT, SHARD
from toolrank.verify_sources import PROVENANCE, archive_path


ATLAS_REVISION = "f8c0e6e8adf0b76a70cc706ad0bb9f96252728eb"
TOUCAN_REVISION = "0df3cf37f2abefb380370cfb02eabea2a35ae782"
TOUCAN_SHARD = "Kimi-K2/train-00000-of-00040.parquet"
ADDITIONAL_SHARDS = {"Kimi-K2": (40, (7, 16, 24)), "OSS": (47, (19,)), "Qwen3": (44, (12,))}


def fetch_archive(job):
    path, url = job
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    result = subprocess.run(["curl", "-fLsS", "--max-time", "45", "--max-filesize", "40000000",
                             "-o", str(partial), url], capture_output=True, check=False)
    if result.returncode:
        partial.unlink(missing_ok=True)
        return f"Archive unavailable: {path.name}"
    partial.replace(path)
    return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance", type=Path, default=PROVENANCE)
    parser.add_argument("--additional-shards", action="store_true")
    args = parser.parse_args()
    etom = f"https://raw.githubusercontent.com/snooow1029/ETOM/{REVISION}/data"
    atlas = f"https://raw.githubusercontent.com/scaleapi/mcp-atlas/{ATLAS_REVISION}"
    toucan = f"https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M/resolve/{TOUCAN_REVISION}"
    jobs = [
        (ROOT / name, f"{etom}/{'queries/' if name.startswith('level_') else ''}{name}")
        for name in FILES
    ]
    jobs += [
        (Path(ATLAS_ROOT), f"{atlas}/services/agent-environment/src/agent_environment/mcp_server_template.json"),
        (Path(SHARD), f"{toucan}/{TOUCAN_SHARD}?download=true"),
    ]
    if args.additional_shards:
        for subset, (count, indices) in ADDITIONAL_SHARDS.items():
            for index in indices:
                name = f"train-{index:05d}-of-{count:05d}.parquet"
                jobs.append((Path(f".work/next/shards/{subset}-{index:05d}.parquet"),
                             f"{toucan}/{subset}/{name}?download=true"))
    archives = []
    for repo, record in read_json(args.provenance).items():
        if record.get("spdx") == "MIT" and record.get("observed_default_head"):
            revision = record["observed_default_head"]
            archives.append((archive_path(repo, revision),
                         f"https://codeload.github.com/{repo}/tar.gz/{revision}"))
    with ThreadPoolExecutor(8) as pool:
        for path in pool.map(download, jobs):
            print(path)
        for path in pool.map(fetch_archive, archives):
            print(path)
