"""Fit and audit the installed gateway on the pinned NWS pilot splits."""
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from evaluate import HERE, read_requests
from examples.openapi_catalog import catalog_from_openapi
from examples.rank_onnx import checkpoint_checksums


def run(args):
    spec = gzip.decompress((HERE / "spec.json.gz").read_bytes())
    spec_hash = sha256(spec).hexdigest()
    protocol = json.loads((HERE / "protocol.json").read_text())
    if spec_hash != protocol["spec_sha256"]:
        raise ValueError("OpenAPI snapshot hash differs from this pilot")
    catalog = catalog_from_openapi(json.loads(spec), "weather")
    names = {tool["name"] for tool in catalog}
    requests_file = HERE / "requests.tsv"
    rows = read_requests(requests_file, names)
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        catalog_path = folder / "catalog.json"
        catalog_path.write_text(json.dumps(catalog))
        base = {"servers": [{"id": "nws", "command": sys.executable,
                            "args": [str(HERE / "upstream.py"), str(catalog_path)], "discover": sorted(names)}]}
        labels = {}
        for split in ("calibration", "holdout"):
            labels[split] = folder / f"{split}.json"
            labels[split].write_text(json.dumps([
                {"query": row["query"], "relevant": ["nws/" + row["expected"]] if row["expected"] else []}
                for row in rows if row["split"] == split
            ]))
        config = folder / "config.json"

        def command(mode, path):
            return [str(args.gateway), str(config), "--checkpoint", str(args.checkpoint), mode, str(path)]

        config.write_text(json.dumps(base))
        fit = json.loads(subprocess.check_output(command("--calibrate", labels["calibration"]), timeout=60))
        if fit["status"] != "calibrated":
            raise ValueError("NWS calibration did not produce a cutoff")
        before = json.loads(subprocess.check_output(command("--evaluate", labels["holdout"]), timeout=60))
        (folder / "cutoff.json").write_text(json.dumps(fit))
        base["calibration_report"] = "cutoff.json"
        config.write_text(json.dumps(base))
        after = json.loads(subprocess.check_output(command("--evaluate", labels["holdout"]), timeout=60))
        unused = min(names - {row["expected"] for row in rows})
        base["servers"][0]["discover"].remove(unused)
        config.write_text(json.dumps(base))
        drift = subprocess.run(command("--evaluate", labels["holdout"]), capture_output=True, text=True, timeout=60)
        if drift.returncode != 2 or "does not match this catalog or search scope" not in drift.stderr:
            raise ValueError("Gateway evaluation accepted a changed catalog under the old cutoff")
        base["servers"][0]["discover"].append(unused)
        config.write_text(json.dumps(base))
        wrong_scorer = fit | {"scope": fit["scope"] | {"scorer_sha256": "0" * 64}}
        (folder / "cutoff.json").write_text(json.dumps(wrong_scorer))
        mismatch = subprocess.run(command("--evaluate", labels["holdout"]), capture_output=True, timeout=60)
        if mismatch.returncode != 2 or b"or calibration report" not in mismatch.stderr:
            raise ValueError("Gateway accepted a cutoff with a different scorer fingerprint")
    return {"scope": "Fixed NWS OpenAPI catalog served over stdio. Calibration and holdout stay separate. No tool calls.",
            "spec_sha256": spec_hash, "requests_sha256": sha256(requests_file.read_bytes()).hexdigest(),
            "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "checkpoint_sha256": checkpoint_checksums(args.checkpoint, "embedding-int8.onnx"), "fit": fit,
            "holdout_without_cutoff": before, "holdout_with_cutoff": after,
            "stale_report_rejected_after_tool_removal": True, "different_scorer_report_rejected": True,
            "external_compute_usd": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args), indent=2) + "\n")
