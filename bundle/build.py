"""Build the desktop bundle from this checkout and the public ToolRank checkpoint."""

import argparse
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parent
MODEL_ID = "christianengel/mcp-toolrank-33m"
REVISION = "a24a121266154006e2bd2e32a41576437c3455c2"
MODEL_FILES = {
    "tokenizer.json": "91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854",
    "sentence_bert_config.json": "65c2293c310f5476a8d1cbada277722c3e7ae5b5ddbaa2d2fe5d075f410d6d02",
    "config_sentence_transformers.json": "7453db62f635567117454bf0599899107d43569c95d5353b5993d809c39379cf",
    "onnx/embedding-int8.onnx": "ba0e544669eac502cac535d3ed53b9cf5b530b5fd929d4a7bddd787abe08ea33",
}
SERVER_FILES = ("search_mcp.py", "rank_onnx.py", "catalog_lexical.py", "catalog_int8.onnx")


def build(output: Path) -> None:
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
            for name, expected_hash in MODEL_FILES.items():
                url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
                with urlopen(url, timeout=90) as source:
                    contents = source.read()
                if sha256(contents).hexdigest() != expected_hash:
                    raise ValueError(f"Checksum mismatch for {name} at {REVISION}")
                add(f"model/{name}", contents)
        archive.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    build(parser.parse_args().output)
