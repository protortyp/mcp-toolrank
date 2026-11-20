"""Download the frozen benchmark inputs without downloading training data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import snapshot_download

from toolrank.baseline_comparison import ETOM_ROOT, MODELS, TOOLRANK_REVISION
from toolrank.etom import FILES, REVISION
from toolrank.fetch import adapt_toolscout, download, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolrank", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--toolret", action="store_true")
    args = parser.parse_args()
    remote = f"https://raw.githubusercontent.com/snooow1029/ETOM/{REVISION}/data"
    jobs = [(ETOM_ROOT / name, f"{remote}/{'queries/' if name.startswith('level_') else ''}{name}")
            for name in FILES]
    targets = [Path(".work/models") / MODELS[name] for name in args.models if name != "toolrank"]
    for path, url in inventory():
        if any(path.is_relative_to(target) for target in targets) or (
            args.toolret and path.is_relative_to(Path(".work/toolret"))
        ):
            jobs.append((path, url))
    with ThreadPoolExecutor(8) as pool:
        for path in pool.map(download, jobs):
            print(path, flush=True)
    if "toolscout" in args.models:
        adapt_toolscout(Path(".work/models") / MODELS["toolscout"])
    if "toolrank" in args.models:
        snapshot_download(MODELS["toolrank"], revision=TOOLRANK_REVISION, token=False,
                          local_dir=args.toolrank, ignore_patterns=["onnx/*", "*.gguf"])


if __name__ == "__main__":
    main()
