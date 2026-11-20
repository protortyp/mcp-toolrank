"""Build the static search bundle or the multi-server gateway bundle."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parent
MODEL = json.loads((BUNDLE / "model.json").read_text())
NOTICE_MANIFEST = BUNDLE / "notices.json"
SERVER_FILES = ("search_mcp.py", "rank_onnx.py", "catalog_lexical.py", "catalog_int8.onnx")


def write_bundle(output, members):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        archive = Path(temp) / output.name
        with ZipFile(archive, "w") as bundle:
            for name, contents in members:
                entry = ZipInfo(name)
                entry.external_attr = 0o100644 << 16
                bundle.writestr(entry, contents, compress_type=ZIP_DEFLATED, compresslevel=6)
        archive.replace(output)


def static_members(checkpoint):
    for name in ("manifest.json", "pyproject.toml", "uv.lock"):
        yield name, (BUNDLE / name).read_bytes()
    yield "LICENSE", (ROOT / "LICENSE").read_bytes()
    for name in SERVER_FILES:
        yield f"src/{name}", (ROOT / "examples" / name).read_bytes()
    notices = json.loads(NOTICE_MANIFEST.read_text())
    yield "model/NOTICES.sha256.json", NOTICE_MANIFEST.read_bytes()
    for name, expected_hash in (MODEL["files"] | notices).items():
        if checkpoint is not None:
            contents = (checkpoint / name).read_bytes()
        else:
            url = f"https://huggingface.co/{MODEL['repo']}/resolve/{MODEL['revision']}/{name}"
            with urlopen(url, timeout=90) as source:
                contents = source.read()
        if sha256(contents).hexdigest() != expected_hash:
            raise ValueError(f"Checksum mismatch for {name} at {MODEL['revision']}")
        yield f"model/{name}", contents


def build(output: Path, checkpoint: Path | None = None) -> None:
    write_bundle(output, static_members(checkpoint))


def gateway_members(wheel):
    gateway = BUNDLE / "gateway"
    lock = (gateway / "uv.lock").read_bytes()
    package = next(row for row in tomllib.loads(lock.decode())["package"] if row["name"] == "toolrank-cpu")
    pinned, = package["wheels"]
    contents = wheel.read_bytes()
    if wheel.name != pinned["filename"] or f"sha256:{sha256(contents).hexdigest()}" != pinned["hash"]:
        raise ValueError("Gateway wheel does not match the bundle lockfile")
    for name in ("manifest.json", "pyproject.toml", "server.py", "example.gateway.json"):
        yield name, (gateway / name).read_bytes()
    yield "uv.lock", lock
    yield "LICENSE", (ROOT / "LICENSE").read_bytes()
    yield package["source"]["path"], contents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--checkpoint", type=Path, help="use a local pinned snapshot with weights and model notices")
    modes.add_argument("--gateway-wheel", type=Path, help="build a gateway bundle with the wheel pinned in its lockfile")
    args = parser.parse_args()
    if args.gateway_wheel is None:
        build(args.output, args.checkpoint)
    else:
        write_bundle(args.output, gateway_members(args.gateway_wheel))
