"""Measure full-request shortlisting and authored intent splitting on 93 tools."""
import argparse
import csv
import gzip
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path
from time import perf_counter

from mcp.types import Tool
import numpy as np
from toolrank_cpu.catalog_lexical import CatalogLexicalIndex
from toolrank_cpu.gateway import Entry, limit_tool_bytes
from toolrank_cpu.rank_onnx import OnnxToolRank, checkpoint_checksums, top_indices

from examples.openapi_catalog import catalog_from_openapi, path_item


HERE = Path(__file__).resolve().parent
NWS = HERE.parent / "nws_pilot" / "spec.json.gz"
INDEPENDENT = Path("data/onnx_int8_holdout.json")
CASES = HERE / "cases.tsv"
PROTOCOL = HERE / "protocol.json"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def schema_bytes(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def reference(document, item):
    ref = item.get("$ref", "")
    if not ref.startswith("#/"):
        return item
    current = document
    for key in ref[2:].split("/"):
        current = current[key.replace("~1", "/").replace("~0", "~")]
    return current


def input_schema(document, operation):
    meta = operation["_meta"]["toolrank/openapi"]
    item = path_item(document, document["paths"][meta["path"]])
    fields = [*item.get("parameters", []), *item[meta["method"].lower()].get("parameters", [])]
    properties, required = {}, []
    for raw in fields:
        field = reference(document, raw)
        if field.get("in") not in ("query", "path"):
            continue
        name = field["name"]
        schema = dict(reference(document, field.get("schema", {})))
        if field.get("description"):
            schema["description"] = field["description"]
        properties[name] = schema
        if field.get("required"):
            required.append(name)
    return {"type": "object", "properties": properties, "required": sorted(set(required))}


def catalog():
    document = json.loads(gzip.decompress(NWS.read_bytes()))
    weather = [dict(operation, repository="nws", inputSchema=input_schema(document, operation))
               for operation in catalog_from_openapi(document, "weather")]
    independent = list(json.loads(INDEPENDENT.read_text())["tools"].values())
    entries = []
    for row in [*weather, *independent]:
        tool = Tool(name=row["name"], description=row["description"], inputSchema=row["inputSchema"])
        revision = sha256(json.dumps([tool.description, tool.inputSchema, tool.outputSchema],
                                     sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
        entries.append(Entry(row["repository"], tool, True, revision))
    return sorted(entries, key=lambda entry: entry.id)


def interleave(first, second, limit):
    result = []
    for pair in zip_longest(first, second):
        for candidate in pair:
            if candidate is not None and candidate not in result:
                result.append(candidate)
                if len(result) == limit:
                    return result
    return result


class Retriever:
    def __init__(self, entries, checkpoints):
        self.entries = entries
        documents = [entry.text for entry in entries]
        self.lexical = CatalogLexicalIndex(documents)
        self.models = {}
        for name, path in checkpoints.items():
            encoder = OnnxToolRank(path, [])
            vectors = np.concatenate([encoder.encode(documents[i:i + 32])
                                      for i in range(0, len(documents), 32)])
            self.models[name] = encoder, vectors

    def rank(self, query):
        tfidf, bm25 = self.lexical.scores(query)
        scores = {"bm25": (None, bm25)}
        for name, (encoder, vectors) in self.models.items():
            cosine = vectors @ encoder.encode([encoder.prefix + query])[0]
            scores[name] = cosine, cosine + .4 * tfidf + .15 * bm25
        return {
            name: [(int(i), None if cosine is None else float(cosine[i]), float(weights[i]))
                   for i in top_indices(weights, 10)]
            for name, (cosine, weights) in scores.items()
        }

    def payload(self, ranking):
        return [self.entries[i].public() | {"cosine": cosine, "ranking_score": score}
                for i, cosine, score in ranking]


def metric(payload, gold, full_size):
    matched = gold & {item["name"] for item in payload}
    return (len(gold), len(matched), len(payload), schema_bytes(payload), full_size)


def summarize(rows):
    data = np.array(rows, dtype=np.int64)
    targets, hits, tools, sizes, full = data.T
    answerable = targets > 0
    return {"requests": len(rows), "answerable": int(answerable.sum()),
            "complete": int(((targets == hits) & answerable).sum()), "required_tool_hits": int(hits.sum()),
            "required_tools": int(targets.sum()), "unsupported_with_suggestions": int((tools[~answerable] > 0).sum()),
            "mean_visible_tools": float(tools.mean()), "tool_json_reduction": float(1 - sizes.sum() / full.sum()),
            "largest_matches_array_bytes": int(sizes.max())}


def run(checkpoints):
    entries = catalog()
    with CASES.open(newline="", encoding="utf-8") as file:
        cases = list(csv.DictReader(file, delimiter="\t"))
    start = perf_counter()
    retriever = Retriever(entries, checkpoints)
    startup = perf_counter() - start
    full = retriever.payload([(i, None, None) for i in range(len(entries))])
    full_size = schema_bytes(full)
    records = {"all": []}
    names = {entry.tool.name for entry in entries}
    joint_ms = []
    for case in cases:
        gold = {case["tool_a"], case["tool_b"]} - {"-"}
        if gold - names or len(gold) == 1:
            raise ValueError(f"Incomplete gold tool labels for {case['id']}")
        records["all"].append(metric(full, gold, full_size))
        start = perf_counter()
        joint = retriever.rank(case["intent_a"] + " " + case["intent_b"])
        joint_ms.append(1000 * (perf_counter() - start))
        first, second = retriever.rank(case["intent_a"]), retriever.rank(case["intent_b"])
        for name, ranking in joint.items():
            split = interleave([i for i, _, _ in first[name]], [i for i, _, _ in second[name]], 5)
            split_scores = {i: (cosine, value) for i, cosine, value in [*second[name], *first[name]]}
            whole = retriever.payload(ranking)
            modes = {
                "joint_top5": whole[:5],
                "joint_top10": whole,
                "joint_top10_4096_bytes": limit_tool_bytes(whole, 4096),
                "split_top5": retriever.payload([(i, *split_scores[i]) for i in split]),
            }
            for method, payload in modes.items():
                records.setdefault(f"{name}_{method}", []).append(metric(payload, gold, full_size))
    scope = {"catalog_tools": len(entries), "providers": sorted({entry.server for entry in entries}),
             "full_matches_array_bytes": full_size}
    return {"scope": scope, "methods": {key: summarize(rows) for key, rows in records.items()},
            "local_startup_index_ms": startup * 1000,
            "joint_query_all_comparators_p95_ms": float(np.quantile(joint_ms[1:], .95)),
            "source_sha256": digest(Path(__file__)), "case_sha256": digest(CASES),
            "protocol_sha256": digest(PROTOCOL), "nws_gzip_sha256": digest(NWS),
            "api_sha256": digest(INDEPENDENT),
            "checkpoint_checksums": {name: checkpoint_checksums(path, "embedding-int8.onnx")
                                     for name, path in checkpoints.items()}, "external_compute_usd": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bge", type=Path, required=True)
    parser.add_argument("--toolrank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run({"bge": args.bge, "toolrank": args.toolrank}), indent=2) + "\n")
