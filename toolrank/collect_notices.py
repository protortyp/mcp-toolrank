"""Stage verified upstream notices for a research checkpoint."""
import argparse
from pathlib import Path
import tarfile

from toolrank.core import read_json, write_json
from toolrank.verify_sources import archive_path, git_blob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    provenance = read_json(args.provenance)
    tools = read_json(args.training)["tools"].values()
    sources = sorted({(tool["repository"], tool["source_commit"]) for tool in tools})
    args.output.mkdir(parents=True, exist_ok=True)
    entries = []
    for repo, revision in sources:
        record = provenance[repo]
        if record["observed_default_head"] != revision:
            raise ValueError(f"training corpus and notice revision differ for {repo}")
        path = record["path"]
        with tarfile.open(archive_path(repo, revision), "r:gz") as archive:
            root = archive.next().name.split("/")[0]
            with archive.extractfile(f"{root}/{path}") as stream:
                notice = stream.read()
        if git_blob(notice) != record["blob"]:
            raise ValueError(f"license content changed for {repo}")
        filename = f"{repo.replace('/', '--')}--{revision[:12]}.txt"
        (args.output / filename).write_bytes(notice)
        entries.append({"repository": repo, "revision": revision, "license": record["spdx"],
                        "license_blob": record["blob"], "notice_file": filename})
    write_json(args.output / "manifest.json", {
        "descriptions": entries,
        "base_model": "https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "query_data": "https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M/tree/0df3cf37f2abefb380370cfb02eabea2a35ae782",
        "query_dataset_declared_license": "Apache-2.0",
    })
    print(f"Staged {len(entries)} upstream source notices")
