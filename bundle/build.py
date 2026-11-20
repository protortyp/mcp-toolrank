"""Build the desktop bundle from this checkout and the public ToolRank checkpoint."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parent
MODEL = json.loads((BUNDLE / "model.json").read_text())
NOTICE_MANIFEST = BUNDLE / "notices.json"
SERVER_FILES = ("search_mcp.py", "rank_onnx.py", "catalog_lexical.py", "catalog_int8.onnx")


def build(output: Path, checkpoint: Path | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        archive = Path(temp) / output.name
        with ZipFile(archive, "w") as bundle:
            def add(name: str, contents: bytes) -> None:
                entry = ZipInfo(name)
                entry.external_attr = 0o100644 << 16
                bundle.writestr(entry, contents, compress_type=ZIP_DEFLATED, compresslevel=6)

            for name in ("manifest.json", "pyproject.toml", "uv.lock"):
                add(name, (BUNDLE / name).read_bytes())
            add("LICENSE", (ROOT / "LICENSE").read_bytes())
            for name in SERVER_FILES:
                add(f"src/{name}", (ROOT / "examples" / name).read_bytes())
            notices = json.loads(NOTICE_MANIFEST.read_text())
            add("model/NOTICES.sha256.json", NOTICE_MANIFEST.read_bytes())
            for name, expected_hash in (MODEL["files"] | notices).items():
                if checkpoint is not None:
                    contents = (checkpoint / name).read_bytes()
                else:
                    url = f"https://huggingface.co/{MODEL['repo']}/resolve/{MODEL['revision']}/{name}"
                    with urlopen(url, timeout=90) as source:
                        contents = source.read()
                if sha256(contents).hexdigest() != expected_hash:
                    raise ValueError(f"Checksum mismatch for {name} at {MODEL['revision']}")
                add(f"model/{name}", contents)
        archive.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="use a local pinned snapshot with weights and model notices")
    args = parser.parse_args()
    build(args.output, args.checkpoint)
