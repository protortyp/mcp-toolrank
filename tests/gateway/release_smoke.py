"""Rebuild, install, and exercise local artifacts outside the checkout."""
import argparse
import asyncio
from datetime import datetime
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


TOOLS = ["clock/current_utc_time", "time_math/minutes_since_midnight"]
CATALOG = [
    {"name": "get_current_time", "description": "Read the current time in a timezone."},
    {"name": "list_exhibits", "description": "Find museum exhibits open today."},
]
CHECKPOINT_ASSETS = ("LICENSE", "model.json", "rank_onnx.py", "catalog_lexical.py", "catalog_int8.onnx")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def run(command, directory):
    uv_cache = subprocess.check_output(["uv", "cache", "dir"], cwd=directory, text=True).strip()
    env = os.environ | {"TOOLRANK_OFFLINE": "1", "XDG_CACHE_HOME": str(directory / "cache"),
                        "UV_CACHE_DIR": uv_cache}
    result = subprocess.run(command, cwd=directory, env=env, check=True, capture_output=True, text=True, timeout=120)
    return result.stdout


async def desktop_bundle(bundle, directory, input_file):
    extracted = directory / "desktop"
    with ZipFile(bundle) as archive:
        notices = json.loads(archive.read("model/NOTICES.sha256.json"))
        source_manifest = Path(__file__).resolve().parents[2] / "bundle/notices.json"
        if notices != json.loads(source_manifest.read_text()):
            raise ValueError("Desktop bundle does not include the pinned model notice manifest")
        if any(sha256(archive.read(f"model/{path}")).hexdigest() != expected for path, expected in notices.items()):
            raise ValueError("Desktop bundle includes altered model notices")
        if "LICENSE" not in notices or "THIRD_PARTY/BAAI-FlagEmbedding.txt" not in notices:
            raise ValueError("Desktop bundle is missing the model or base-model license")
        archive.extractall(extracted)
    manifest = json.loads((extracted / "manifest.json").read_text())
    config = manifest["server"]["mcp_config"]
    params = [part.replace("${__dirname}", str(extracted)).replace("${user_config.catalog}", str(input_file))
              for part in config["args"]]
    params.insert(1, "--offline")
    async with stdio_client(StdioServerParameters(command=config["command"], args=params, cwd=str(directory))) as streams:
        async with ClientSession(*streams) as client:
            await client.initialize()
            methods = [tool.name for tool in (await client.list_tools()).tools]
            if methods != ["search_tools"]:
                raise ValueError("Unexpected desktop bundle tools")
            result = await client.call_tool("search_tools", {"query": "What time is it now?", "limit": 1})
            matches = (result.structuredContent or {}).get("matches", [])
            if result.isError or not matches or matches[0]["tool"]["name"] != "get_current_time":
                raise ValueError("Desktop bundle returned an unexpected match")
    return {"version": manifest["version"], "search_match": "get_current_time", "model_notices_verified": len(notices)}


async def check(args):
    with TemporaryDirectory() as temp:
        directory = Path(temp)
        release = [args.wheel, args.sdist, args.bundle]
        run(["uv", "build", "--wheel", "--offline", str(args.sdist), "--out-dir", str(directory / "rebuild")], directory)
        rebuilt = directory / "rebuild" / args.wheel.name
        if digest(rebuilt) != digest(args.wheel):
            raise ValueError("Wheel rebuilt from the isolated source distribution differs")
        with ZipFile(args.wheel) as wheel:
            if any(f"toolrank_cpu/{name}" not in wheel.namelist() for name in CHECKPOINT_ASSETS):
                raise ValueError("Wheel is missing an expected runtime asset")
        records = []
        for python in ("3.12", "3.13"):
            start = perf_counter()
            demo = json.loads(run(["uvx", "--isolated", "--offline", "--python", python,
                                   "--from", f"{args.wheel}[gateway]", "toolrank-gateway", "--demo"], directory))
            elapsed = perf_counter() - start
            utc = datetime.fromisoformat(demo["timestamp"])
            if not (demo["ranking"] == "bm25" and demo["discovered_tools"] == TOOLS == demo["invoked_tools"]
                    and demo["minutes_since_midnight"] == utc.hour * 60 + utc.minute):
                raise ValueError("Installed gateway demo returned unexpected results")
            if (directory / "cache/toolrank").exists():
                raise ValueError("The model-free demo touched the checkpoint cache")
            records.append({"python": python, "elapsed_seconds": elapsed, "ordered_calls": demo["invoked_tools"],
                            "minutes_match_clock_receipt": True, "empty_model_cache": True})
        catalog = directory / "catalog.json"
        catalog.write_text(json.dumps(CATALOG))
        ranked = json.loads(run(["uvx", "--isolated", "--offline", "--python", "3.12", "--from", str(args.wheel),
                                 "toolrank-cpu", str(catalog), "What time is it now?", "--checkpoint",
                                 str(args.checkpoint), "--hybrid", "--limit", "1"], directory))
        if ranked[0]["tool"]["name"] != "get_current_time":
            raise ValueError("Installed CPU ranker returned an unexpected match")
        desktop = await desktop_bundle(args.bundle, directory, catalog)
    return {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": digest(Path(__file__)), "platform": platform.platform(),
            "uv_version": subprocess.check_output(["uv", "--version"], text=True).strip(),
            "artifacts": {path.name: digest(path) for path in release}, "wheel_rebuild_identical": True,
            "offline_with_cached_dependencies": True, "fresh_install_demos": records,
            "installed_onnx_cli_match": ranked[0]["tool"]["name"], "desktop_bundle": desktop,
            "external_compute_usd": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("wheel", "sdist", "bundle", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=lambda value: Path(value).resolve(), required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(asyncio.run(check(args)), indent=2) + "\n")
