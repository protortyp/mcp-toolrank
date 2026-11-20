"""Record licenses at immutable commits for candidate upstream repositories."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import subprocess

from toolrank.core import read_json, write_json


def github(path, selector):
    result = subprocess.run(["gh", "api", path, "--jq", selector],
                            capture_output=True, text=True, timeout=25, check=False)
    if result.returncode or not result.stdout.strip():
        return None
    return result.stdout.strip()


def inspect(repo):
    if not re.fullmatch(r"[a-z0-9_.-]+/[a-z0-9_.-]+", repo):
        raise ValueError(f"invalid repository ID: {repo}")
    revision = github(f"repos/{repo}/commits?per_page=1", ".[0].sha")
    if not revision or not re.fullmatch(r"[a-f0-9]{40}", revision):
        return repo, {"unavailable": True}
    data = github(f"repos/{repo}/license?ref={revision}",
                  "{spdx: .license.spdx_id, path: .path, blob: .sha}")
    evidence = json.loads(data) if data else {"unavailable": True}
    evidence["observed_default_head"] = revision
    return repo, evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default=".work/next/source-inventory.json")
    parser.add_argument("--output", default=".work/next/expanded-source-licenses.json")
    arguments = parser.parse_args()
    repos = read_json(arguments.inventory)
    with ThreadPoolExecutor(12) as pool:
        report = dict(pool.map(inspect, repos))
    write_json(arguments.output, report)
    print(f"{len(report)} repositories, {sum(row.get('spdx') == 'MIT' for row in report.values())} dated MIT records")
