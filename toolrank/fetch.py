"""Mirror the exact public model and dataset revisions used by the pilot."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import urllib.request


REVISIONS = {
    "datasets/mangopy/ToolRet-Queries": "b8c76ad3349ff17497b6bdb28bb5b8f61a0f6445",
    "datasets/mangopy/ToolRet-Tools": "e06c38c75612b6536bd959e08cdd345894aba6a7",
    "models/BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "models/nuhmanpk/toolscout-bge-small": "7d8e0415d86d896bf2330d7c3e5df7f73744f352",
    "models/TaylorAI/bge-micro-v2": "3edf6d7de0faa426b09780416fe61009f26ae589",
    "models/Lux1997/Tool-Embed-0.6B": "103d16d3593dec6c6f2217be620febb846932f6a",
}


def inventory():
    for repo, revision in REVISIONS.items():
        kind, model_id = repo.split("/", 1)
        base = Path(".work/toolret") if kind == "datasets" else Path(".work/models") / model_id
        api_url = f"https://huggingface.co/api/{repo}/revision/{revision}"
        with urllib.request.urlopen(api_url, timeout=30) as response:
            files = json.load(response)["siblings"]
        for entry in files:
            file = Path(entry["rfilename"])
            needed = file.suffix == ".parquet" if kind == "datasets" else (
                file.suffix in {".json", ".txt", ".safetensors", ".md"} and "onnx" not in file.parts
            )
            if needed:
                repo_path = repo if kind == "datasets" else model_id
                yield base / file, f"https://huggingface.co/{repo_path}/resolve/{revision}/{file}?download=true&full=1"


def download(job):
    destination, url = job
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        subprocess.run(
            ["curl", "-fLsS", "--connect-timeout", "10", "--max-time", "240", "--retry", "2",
             "-o", str(partial), url], check=True,
        )
        partial.replace(destination)
    return destination


def adapt_toolscout(root):
    # The checkpoint uses the 5.4 path and constructor names for these built-ins.
    module_path = root / "modules.json"
    modules = json.loads(module_path.read_text())
    names = [module["type"].rsplit(".", 1)[-1] for module in modules]
    if names != ["Transformer", "Pooling", "Normalize"]:
        raise ValueError(f"unexpected ToolScout module layout: {names}")
    for module, name in zip(modules, names):
        module["type"] = f"sentence_transformers.models.{name}"
    module_path.write_text(json.dumps(modules, indent=2) + "\n")

    config_path = root / "sentence_bert_config.json"
    config = json.loads(config_path.read_text())
    if config.get("transformer_task", "feature-extraction") != "feature-extraction":
        raise ValueError("ToolScout is expected to be a text feature extractor")
    for key in ("transformer_task", "modality_config", "module_output_name"):
        config.pop(key, None)
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    pooling_path = root / "1_Pooling" / "config.json"
    pooling = json.loads(pooling_path.read_text())
    if "embedding_dimension" in pooling:
        pooling["word_embedding_dimension"] = pooling.pop("embedding_dimension")
    pooling_path.write_text(json.dumps(pooling, indent=2) + "\n")


if __name__ == "__main__":
    with ThreadPoolExecutor(12) as pool:
        for path in pool.map(download, inventory()):
            print(path, flush=True)
    adapt_toolscout(Path(".work/models/nuhmanpk/toolscout-bge-small"))
