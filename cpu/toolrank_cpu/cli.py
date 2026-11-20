"""Run the released ToolRank ONNX ranker with a verified local model cache."""

from hashlib import file_digest
import json
import os
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from . import rank_onnx


def matches(path, expected):
    try:
        with path.open("rb") as stream:
            return file_digest(stream, "sha256").hexdigest() == expected
    except FileNotFoundError:
        return False


def checkpoint():
    model = json.loads(Path(__file__).with_name("model.json").read_text(encoding="utf-8"))
    cache_base = Path(os.environ.get("XDG_CACHE_HOME") or os.environ.get("LOCALAPPDATA") or Path.home() / ".cache")
    directory = cache_base / "toolrank" / model["revision"]
    offline = os.environ.get("TOOLRANK_OFFLINE") == "1"
    for name, digest in model["files"].items():
        destination = directory / name
        if matches(destination, digest):
            continue
        if offline:
            raise ValueError(f"Offline mode needs a verified cached file at {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent) as staging:
            temporary = Path(staging) / destination.name
            address = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{name}"
            with urlopen(address, timeout=90) as response, temporary.open("wb") as stream:
                copyfileobj(response, stream)
            if not matches(temporary, digest):
                raise ValueError(f"ToolRank checkpoint checksum mismatch for {name}")
            temporary.replace(destination)
    return directory


def main():
    rank_onnx.main(resolve_checkpoint=checkpoint)


def gateway_main():
    try:
        from .gateway import main as run_gateway
    except ModuleNotFoundError as error:
        if error.name != "mcp":
            raise
        raise SystemExit("Install toolrank-cpu with the gateway extra.") from None
    run_gateway()
