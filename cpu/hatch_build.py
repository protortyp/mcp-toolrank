"""Include shared runtime assets in both the wheel and source archive."""
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


ASSETS = {
    "LICENSE": "LICENSE",
    "rank_onnx.py": "examples/rank_onnx.py",
    "catalog_lexical.py": "examples/catalog_lexical.py",
    "catalog_int8.onnx": "examples/catalog_int8.onnx",
    "model.json": "bundle/model.json",
}


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root)
        for name, source in ASSETS.items():
            dest = f"toolrank_cpu/{name}"
            # The source distribution already contains these assets.
            if not (root / dest).is_file():
                build_data["force_include"][str(root.parent / source)] = dest
