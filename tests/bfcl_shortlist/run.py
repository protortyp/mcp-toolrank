"""Measure CPU tool shortlists on pinned BFCL native menus."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime

from examples.catalog_lexical import CatalogLexicalIndex
from examples.rank_onnx import OnnxToolRank, checkpoint_checksums, top_indices


ROOT = Path(__file__).resolve().parent
NWS_RESULTS = {"bge": Path("results/nws_bge_pilot.json"), "toolrank": Path("results/nws_pilot.json")}
BUDGET = 5
SEED = 20261120


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def json_length(items):
    return len(json.dumps(items, separators=(",", ":"), ensure_ascii=False).encode())


@dataclass
class Case:
    subset: str
    query: str
    tools: list
    gold: set

    @property
    def size_band(self):
        count = len(self.tools)
        if count == 0:
            return "0"
        if count <= 5:
            return "1–5"
        if count <= 10:
            return "6–10"
        return "11–20" if count <= 20 else "21+"


def messages(question):
    content = [row["content"] for turn in question for row in turn if row["role"] == "user"]
    if not content or any(not isinstance(text, str) or not text.strip() for text in content):
        raise ValueError("Expected nonempty textual user messages")
    return "\n".join(content)


def load(root, manifest):
    if (root / "REVISION").read_text().strip() != manifest["revision"]:
        raise ValueError("BFCL revision differs from the frozen manifest")
    cases = []
    for subset, metadata in manifest["datasets"].items():
        file = f"BFCL_v4_{subset}.json"
        path = root / file
        if digest(path) != metadata["data_sha256"]:
            raise ValueError(f"Unexpected benchmark data for {subset}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        gold_by_id = {}
        if "gold_sha256" in metadata:
            gold_file = root / "possible_answer" / file
            if digest(gold_file) != metadata["gold_sha256"]:
                raise ValueError(f"Unexpected gold labels for {subset}")
            for row in map(json.loads, gold_file.read_text().splitlines()):
                gold_by_id[row["id"]] = {name for call in row["ground_truth"] for name in call}
            if set(gold_by_id) != {row["id"] for row in rows}:
                raise ValueError(f"Gold labels do not join for {subset}")
        for row in rows:
            tools = sorted(row["function"], key=lambda tool: tool["name"])
            names = {tool["name"] for tool in tools}
            gold = gold_by_id.get(row["id"], set())
            if len(names) != len(tools) or gold - names or (gold_by_id and not gold):
                raise ValueError(f"Invalid catalog or gold labels for {subset}")
            cases.append(Case(subset, messages(row["question"]), tools, gold))
    return cases


def description(tool):
    return f"{tool['name']}: {tool['description']}"


def embed(ranker, texts):
    return np.concatenate([ranker.encode(texts[offset:offset + 32]) for offset in range(0, len(texts), 32)])


def select(catalog, ordered, budget=BUDGET, min_savings=None):
    kept = ordered[:budget]
    if min_savings is not None:
        shortlist_bytes = json_length([catalog[index] for index in kept])
        if shortlist_bytes > (1 - min_savings) * json_length(catalog):
            return ordered
    return kept


def measure(catalog, gold, indices, full_bytes):
    selected = [catalog[index] for index in indices]
    names = {tool["name"] for tool in selected}
    return (bool(gold), bool(selected), bool(gold) and gold <= names, bool(gold & names),
            len(selected), json_length(selected), full_bytes)


def aggregate(rows):
    data = np.asarray(rows, dtype=np.int64)
    answerable, exposed, complete, any_hit, count, kept_bytes, total_bytes = data.T
    yes = int(answerable.sum())
    no = len(rows) - yes
    return {"requests": len(rows), "answerable": yes, "unsupported": no,
            "complete": int(complete.sum()), "hit": int(any_hit.sum()),
            "accepted_unsupported": int(exposed[answerable == 0].sum()),
            "accepted_supported": int(exposed[answerable == 1].sum()),
            "mean_visible_tools": float(count.mean()),
            "tool_json_bytes": int(total_bytes.sum()), "shortlist_json_bytes": int(kept_bytes.sum()),
            "tool_json_reduction": float(1 - kept_bytes.sum() / total_bytes.sum())}


def answerable_ceiling(cases):
    rows = defaultdict(list)
    for case in cases:
        if case.gold:
            selected = [i for i, tool in enumerate(case.tools) if tool["name"] in case.gold]
            value = measure(case.tools, case.gold, selected, json_length(case.tools))
            rows["overall"].append(value)
            rows[case.size_band].append(value)
    return {band: aggregate(values) for band, values in rows.items()}


def interval(target, control):
    differences = np.array([int(a[2]) - int(b[2]) for a, b in zip(target, control, strict=True) if a[0]])
    samples = np.random.default_rng(SEED).choice(differences, size=(3000, len(differences)), replace=True)
    return {"difference": float(differences.mean()),
            "ci95": np.quantile(samples.mean(axis=1), [.025, .975]).tolist(),
            "answerable_pairs": len(differences)}


def compare(cases, checkpoints):
    documents = sorted({description(tool) for case in cases for tool in case.tools})
    positions = {text: index for index, text in enumerate(documents)}
    vectors = {}
    timings = {}
    thresholds = {}
    for name, checkpoint in checkpoints.items():
        model = OnnxToolRank(checkpoint, [])
        start = perf_counter()
        indexed = embed(model, documents)
        indexing_ms = 1000 * (perf_counter() - start)
        start = perf_counter()
        queries = embed(model, [model.prefix + case.query for case in cases])
        encoding_ms = 1000 * (perf_counter() - start)
        vectors[name] = indexed, queries
        timings[name] = {"cached_documents": len(indexed), "batch_index_ms": indexing_ms,
                         "batch_query_encoding_ms": encoding_ms}
        thresholds[name] = json.loads(NWS_RESULTS[name].read_text())["calibration"]["cosine"]["min_score"]
    metric_rows = defaultdict(lambda: defaultdict(list))
    for index, case in enumerate(cases):
        tools = case.tools
        descriptions = [description(tool) for tool in tools]
        tfidf, bm25 = (CatalogLexicalIndex(descriptions).scores(case.query) if descriptions
                       else (np.empty(0), np.empty(0)))
        ids = np.asarray([positions[text] for text in descriptions], dtype=np.intp)
        full_bytes = json_length(tools)
        scores = {"all": np.zeros(len(ids)), "bm25": bm25}
        accepted = {}
        for name, (docs, queries) in vectors.items():
            cosine = docs[ids] @ queries[index]
            scores[f"{name}_dense"] = cosine
            scores[f"{name}_hybrid"] = cosine + .4 * tfidf + .15 * bm25
            accepted[name] = cosine >= thresholds[name]
        for name, values in scores.items():
            order = top_indices(values, len(values))
            exposures = {"all": order} if name == "all" else {
                f"{name}_top5": select(tools, order),
                f"{name}_top10": select(tools, order, budget=10),
                f"{name}_half_bytes": select(tools, order, min_savings=.5),
            }
            if name.endswith("_hybrid"):
                valid = accepted[name.removesuffix("_hybrid")]
                exposures[f"{name}_nws_cutoff_top5"] = order[valid[order]][:5]
            for policy, shortlist in exposures.items():
                stats = measure(tools, case.gold, shortlist, full_bytes)
                for band in ("overall", case.size_band, case.subset):
                    metric_rows[band][policy].append(stats)
    output = {}
    for band, methods in metric_rows.items():
        output[band] = {"metrics": {method: aggregate(rows) for method, rows in methods.items()}}
        if any(row[0] for row in methods["all"]):
            output[band]["paired_complete_hybrid_top5"] = interval(
                methods["toolrank_hybrid_top5"], methods["bge_hybrid_top5"])
    return output, timings, thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfcl", type=Path, required=True)
    parser.add_argument("--bge", type=Path, required=True)
    parser.add_argument("--toolrank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    cases = load(args.bfcl, manifest)
    checkpoints = {"bge": args.bge, "toolrank": args.toolrank}
    output, timing, thresholds = compare(cases, checkpoints)
    result = {"manifest": manifest, "sample_count": len(cases), "checkpoint_checksums": {
                  name: checkpoint_checksums(path, "embedding-int8.onnx") for name, path in checkpoints.items()
              }, "cpu_threads": 4, "onnxruntime": onnxruntime.__version__, "timing": timing,
              "source_sha256": digest(Path(__file__)), "protocol_sha256": digest(ROOT / "protocol.json"),
              "nws_rejection_cutoffs": thresholds, "paired_interval_seed": SEED,
              "groups": output, "answerable_gold_only_byte_ceiling": answerable_ceiling(cases),
              "external_compute_usd": 0}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
