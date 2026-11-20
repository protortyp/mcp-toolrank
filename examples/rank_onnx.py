# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.5", "onnxruntime==1.24.1", "tokenizers==0.22.2"]
# ///
"""Rank an MCP catalog with CPU ONNX Runtime."""
import argparse
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
from zipfile import BadZipFile

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer, __version__ as tokenizers_version


def quantize_rows(matrix, bound):
    peaks = np.abs(matrix).max(axis=1, keepdims=True)
    scales = np.where(peaks > 0, peaks / bound, np.float32(1))
    return np.rint(matrix / scales).astype(np.int8), scales[:, 0]


def top_indices(scores, limit):
    if limit >= len(scores):
        return np.argsort(-scores, kind="stable")
    cutoff = np.partition(scores, len(scores) - limit)[-limit]
    higher = np.flatnonzero(scores > cutoff)
    ties = np.flatnonzero(scores == cutoff)[:limit - len(higher)]
    return np.concatenate((higher[np.argsort(-scores[higher], kind="stable")], ties))


def encode_compact_catalog(batches, size):
    scales = np.empty(size, dtype=np.float32)
    vectors = np.empty((0, 0), dtype=np.int8)
    offset = 0
    for batch in batches:
        if offset == 0:
            vectors = np.empty((batch.shape[1], size), dtype=np.int8)
        # With a U8 query, two products at these limits fit in a signed 16-bit lane.
        encoded, batch_scales = quantize_rows(batch, 63)
        end = offset + len(batch)
        vectors[:, offset:end] = encoded.T
        scales[offset:end] = batch_scales
        offset = end
    return vectors, scales


class CompactCatalog:
    def __init__(self, batches, size, threads, *, cached=None):
        self.vectors, self.scales = cached if cached is not None else encode_compact_catalog(batches, size)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(
            str(Path(__file__).with_name("catalog_int8.onnx")), sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def scores(self, queries):
        encoded, scales = quantize_rows(queries, 127)
        unsigned = (encoded.astype(np.int16) + 128).astype(np.uint8)
        dots = self.session.run(["scores"], {"query": unsigned, "catalog": self.vectors})[0]
        return dots.astype(np.float32) * self.scales * scales[:, None]


def catalog_cache_path(directory, checkpoint, descriptions, model_file, precision, threads):
    scope = {
        "checkpoint": checkpoint_checksums(checkpoint, model_file),
        "descriptions": descriptions,
        "catalog_precision": precision,
        "threads": threads,
        "onnxruntime": ort.__version__,
        "tokenizers": tokenizers_version,
    }
    key = sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
    return directory / f"toolrank-v1-{key}.npz"


def read_cached_vectors(path, size, dimension, precision):
    try:
        with path.open("rb") as source, np.load(source, allow_pickle=False) as archive:
            if str(archive["key"]) != path.stem:
                return None
            vectors = archive["vectors"]
            if precision == "int8":
                scales = archive["scales"]
                if (vectors.shape == (dimension, size) and vectors.dtype == np.int8
                        and scales.shape == (size,) and scales.dtype == np.float32
                        and np.isfinite(scales).all() and (scales > 0).all()):
                    return vectors, scales
            elif (vectors.shape == (size, dimension) and vectors.dtype == np.float32
                  and np.isfinite(vectors).all()):
                return vectors, None
    except (FileNotFoundError, ValueError, TypeError, KeyError, EOFError, BadZipFile):
        pass
    return None


def write_cached_vectors(path, arrays):
    vectors, scales = arrays
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            fields = {"key": np.array(path.stem), "vectors": vectors}
            if scales is not None:
                fields["scales"] = scales
            np.savez(stream, **fields)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


class OnnxToolRank:
    def __init__(self, checkpoint, tools, *, threads=4, model_file="embedding-int8.onnx",
                 catalog_precision="float32", hybrid=False, cache_dir=None):
        if catalog_precision not in ("float32", "int8"):
            raise ValueError("catalog_precision must be float32 or int8")
        self.tools = sorted(tools, key=lambda item: item["name"])
        if len({item["name"] for item in self.tools}) != len(self.tools):
            raise ValueError("catalog tool names must be unique")
        self.tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
        cfg = json.loads((checkpoint / "sentence_bert_config.json").read_text())
        self.tokenizer.enable_truncation(max_length=cfg["max_seq_length"])
        pad_id = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
        prompts = json.loads((checkpoint / "config_sentence_transformers.json").read_text())
        self.prefix = prompts["prompts"]["query"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(checkpoint / "onnx" / model_file), sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.inputs = [item.name for item in self.session.get_inputs()]
        descriptions = [f"{item['name']}: {item.get('description') or ''}" for item in self.tools]
        self.lexical_index = None
        if hybrid and descriptions:
            if __package__:
                from .catalog_lexical import CatalogLexicalIndex
            else:
                from catalog_lexical import CatalogLexicalIndex
            self.lexical_index = CatalogLexicalIndex(descriptions)
        batches = (self.encode(descriptions[i:i + 32]) for i in range(0, len(descriptions), 32))
        self.catalog = None
        self.docs = np.empty((0, 384), dtype=np.float32)
        if not descriptions:
            return
        size = len(descriptions)
        cached = cache_path = None
        if cache_dir is not None:
            cache_path = catalog_cache_path(cache_dir, checkpoint, descriptions, model_file,
                                            catalog_precision, threads)
            dimension = next(output.shape[-1] for output in self.session.get_outputs() if output.name == "embedding")
            cached = read_cached_vectors(cache_path, size, dimension, catalog_precision)
        if catalog_precision == "int8":
            self.catalog = CompactCatalog(batches, size, threads, cached=cached)
            self.docs = None
            built = self.catalog.vectors, self.catalog.scales
        else:
            self.docs = cached[0] if cached is not None else np.concatenate(list(batches))
            built = self.docs, None
        if cache_path is not None and cached is None:
            write_cached_vectors(cache_path, built)

    def encode(self, texts):
        rows = self.tokenizer.encode_batch(texts)
        tokens = {
            "input_ids": [row.ids for row in rows],
            "attention_mask": [row.attention_mask for row in rows],
            "token_type_ids": [row.type_ids for row in rows],
        }
        return self.session.run(["embedding"],
                                {name: np.array(tokens[name], dtype=np.int64) for name in self.inputs})[0]

    def cosine_scores(self, request):
        query = self.encode([self.prefix + request])
        return self.catalog.scores(query)[0] if self.catalog else self.docs @ query[0]

    def rank(self, request, *, limit=5, min_score=None):
        if limit < 0:
            raise ValueError("limit must be nonnegative")
        if not request.strip() or not self.tools or not limit:
            return []
        cosine = self.cosine_scores(request)
        scores = self.lexical_index.blend(request, cosine) if self.lexical_index else cosine
        if min_score is None:
            order = top_indices(scores, limit)
        else:
            eligible = np.flatnonzero(cosine >= min_score)
            order = eligible[top_indices(scores[eligible], limit)]
        return [{"tool": self.tools[index], "score": float(cosine[index])} for index in order]


def read_catalog(data):
    if isinstance(data, dict) and "result" in data:
        data = data["result"]
    tools = data.get("tools") if isinstance(data, dict) else data
    if not isinstance(tools, list) or any(
        not isinstance(tool, dict) or not isinstance(tool.get("name"), str) or not tool["name"].strip()
        for tool in tools
    ):
        raise ValueError("catalog must contain a list of tool definitions with nonempty names")
    return tools


def read_cases(data, names):
    if not isinstance(data, list) or not data:
        raise ValueError("evaluation labels must be a nonempty JSON array")
    cases = []
    for index, entry in enumerate(data, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"case {index} must be an object")
        request, expected = entry.get("query"), entry.get("relevant")
        if not isinstance(request, str) or not request.strip():
            raise ValueError(f"case {index} needs a nonempty query")
        if not isinstance(expected, list) or any(not isinstance(name, str) for name in expected):
            raise ValueError(f"case {index} needs a relevant array of tool names")
        if unknown := set(expected) - names:
            raise ValueError(f"case {index} names tools outside the catalog: {', '.join(sorted(unknown))}")
        cases.append((request, set(expected)))
    return cases


def evaluate(retriever, cases, *, limit, min_score):
    discounts = [1 / math.log2(rank + 2) for rank in range(limit)]
    totals = dict.fromkeys(("hit", "recall", "ndcg"), 0.0)
    supported = unsupported = accepted_supported = accepted_unsupported = 0
    durations = []
    retriever.rank(cases[0][0], limit=limit)
    for request, expected in cases:
        start = time.perf_counter()
        matches = retriever.rank(request, limit=limit)
        durations.append((time.perf_counter() - start) * 1_000)
        accepted = bool(matches)
        if min_score is not None and (not matches or matches[0]["score"] < min_score):
            accepted = bool(retriever.rank(request, limit=1, min_score=min_score))
        if not expected:
            unsupported += 1
            accepted_unsupported += accepted
            continue
        supported += 1
        accepted_supported += accepted
        hits = [match["tool"]["name"] in expected for match in matches]
        totals["hit"] += any(hits)
        totals["recall"] += sum(hits) / len(expected)
        totals["ndcg"] += sum(hit * weight for hit, weight in zip(hits, discounts)) / sum(
            discounts[:len(expected)]
        )
    return {
        "k": limit,
        "min_score": min_score,
        "supported": supported,
        "unsupported": unsupported,
        "metrics": {f"{name}@{limit}": value / supported if supported else None
                    for name, value in totals.items()},
        "accepted_supported": accepted_supported,
        "accepted_unsupported": accepted_unsupported,
        "supported_coverage": accepted_supported / supported if supported else None,
        "unsupported_false_accept_rate": accepted_unsupported / unsupported if unsupported else None,
        "query_p50_ms": float(np.median(durations)),
        "query_p95_ms": float(np.quantile(durations, 0.95)),
    }


def allowed_negative_exceedances(count, max_false_accept, confidence):
    """Return the largest exact-binomial order statistic supporting the risk target."""
    log_failure = math.log1p(-confidence)
    log_p = count * math.log1p(-max_false_accept)
    if log_p > log_failure:
        return None
    cumulative = log_p
    log_odds = math.log(max_false_accept) - math.log1p(-max_false_accept)
    for exceedances in range(1, count + 1):
        log_p += math.log(count - exceedances + 1) - math.log(exceedances) + log_odds
        cumulative = float(np.logaddexp(cumulative, log_p))
        if cumulative > log_failure:
            return exceedances - 1
    return count - 1


def calibrate(cases, *, score_requests, max_false_accept, confidence):
    if not 0 < max_false_accept < 1 or not 0 < confidence < 1:
        raise ValueError("max_false_accept and confidence must be between zero and one")
    negative_count = sum(not expected for _, expected in cases)
    positive_count = len(cases) - negative_count
    order = allowed_negative_exceedances(negative_count, max_false_accept, confidence)
    report = {
        "status": "insufficient_no_match_cases" if order is None else "calibrated",
        "max_false_accept": max_false_accept, "confidence": confidence,
        "no_match_cases": negative_count, "answerable_cases": positive_count,
        "min_score": None,
    }
    if order is None:
        required = math.log1p(-confidence) / math.log1p(-max_false_accept)
        report["minimum_no_match_cases"] = math.ceil(required) if math.isfinite(required) else None
        return report

    # The cutoff is strictly above the order statistic, including when scores tie.
    scores = np.asarray(score_requests([request for request, _ in cases]), dtype=np.float32)
    if scores.shape != (len(cases),) or not np.isfinite(scores).all():
        raise ValueError("calibration requires one finite score per request")
    answerable = np.array([bool(expected) for _, expected in cases])
    descending = np.sort(scores[~answerable])[::-1]
    threshold = np.nextafter(descending[order], np.float32(np.inf))
    if not np.isfinite(threshold):
        raise ValueError("could not represent a finite cutoff")
    accepted = scores >= threshold
    report.update(
        min_score=float(threshold),
        order_statistic_rank=negative_count - order,
        accepted_no_match_on_calibration=int((accepted & ~answerable).sum()),
        accepted_answerable_on_calibration=int((accepted & answerable).sum()),
    )
    return report


def checkpoint_checksums(checkpoint, model_file):
    files = ["tokenizer.json", "sentence_bert_config.json", "config_sentence_transformers.json",
             f"onnx/{model_file}"]
    checksums = {}
    for file in files:
        with (checkpoint / file).open("rb") as stream:
            checksums[file] = file_digest(stream, "sha256").hexdigest()
    return checksums


def calibration_scope(checkpoint, tool_catalog, model_file, catalog_precision, threads):
    canonical = json.dumps(sorted(tool_catalog, key=lambda tool: tool["name"]), sort_keys=True)
    return {
        "catalog_sha256": sha256(canonical.encode()).hexdigest(),
        "checkpoint_sha256": checkpoint_checksums(checkpoint, model_file),
        "catalog_precision": catalog_precision,
        "threads": threads, "onnxruntime": ort.__version__,
        "score": "maximum embedding cosine",
    }


def main(*, resolve_checkpoint=None):
    parser = argparse.ArgumentParser()
    if resolve_checkpoint is None:
        parser.add_argument("checkpoint", type=Path)
    else:
        parser.add_argument("--checkpoint", type=Path, help="use a local checkpoint folder")
    parser.add_argument("catalog", type=Path, help="an MCP tools/list result or an array of tool definitions")
    parser.add_argument("request", nargs="?")
    parser.add_argument("--evaluate", type=Path, metavar="LABELS", help="measure on labeled requests")
    parser.add_argument("--calibrate", type=Path, metavar="LABELS", help="fit a cutoff on separate calibration cases")
    parser.add_argument("--max-false-accept", type=float, default=.10)
    parser.add_argument("--confidence", type=float, default=.95)
    parser.add_argument("--model-file", default="embedding-int8.onnx")
    parser.add_argument("--compare-model-file", help="evaluate a second ONNX graph over the same requests")
    parser.add_argument("--hybrid", action="store_true", help="add fixed TF-IDF and name-weighted BM25 scores")
    parser.add_argument("--catalog-cache", type=Path, metavar="DIR", help="reuse encoded catalog vectors from disk")
    parser.add_argument("--catalog-precision", choices=["float32", "int8"], default="float32")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if sum(value is not None for value in (args.request, args.evaluate, args.calibrate)) != 1:
        parser.error("provide a request, --evaluate LABELS, or --calibrate LABELS")
    if args.calibrate and args.min_score is not None:
        parser.error("--calibrate determines min_score")
    if not 0 < args.max_false_accept < 1 or not 0 < args.confidence < 1:
        parser.error("--max-false-accept and --confidence must be between zero and one")
    if args.threads < 1 or args.limit < 0 or (args.evaluate and args.limit == 0):
        parser.error("--threads must be positive and --limit must be positive for evaluation")
    if args.min_score is not None and not math.isfinite(args.min_score):
        parser.error("--min-score must be finite")
    if args.compare_model_file and (not args.evaluate or args.compare_model_file == args.model_file):
        parser.error("--compare-model-file requires evaluation and a different ONNX graph")
    try:
        tools = read_catalog(json.loads(args.catalog.read_text()))
        labels = args.evaluate or args.calibrate
        if labels:
            cases = read_cases(json.loads(labels.read_text()), {tool["name"] for tool in tools})
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if args.calibrate and not tools:
        parser.error("calibration requires a nonempty tool catalog")
    if args.checkpoint is None:
        try:
            args.checkpoint = resolve_checkpoint()
        except (OSError, ValueError) as error:
            parser.error(str(error))

    def load(model_file):
        return OnnxToolRank(args.checkpoint, tools, threads=args.threads, model_file=model_file,
                            catalog_precision=args.catalog_precision, hybrid=args.hybrid and args.calibrate is None,
                            cache_dir=args.catalog_cache)

    if args.calibrate:
        def score_requests(requests):
            retriever = load(args.model_file)
            return [retriever.cosine_scores(request).max() for request in requests]

        try:
            result = calibrate(cases, score_requests=score_requests, max_false_accept=args.max_false_accept,
                               confidence=args.confidence)
            if result["min_score"] is not None:
                result["scope"] = calibration_scope(
                    args.checkpoint, tools, args.model_file, args.catalog_precision, args.threads
                )
        except (OSError, ValueError) as error:
            parser.error(str(error))
    elif args.evaluate:
        models = [args.model_file]
        if args.compare_model_file:
            models.append(args.compare_model_file)
        result = {
            "catalog_size": len(tools),
            "threads": args.threads,
            "catalog_precision": args.catalog_precision,
            "hybrid": args.hybrid,
            "models": {name: evaluate(load(name), cases, limit=args.limit, min_score=args.min_score)
                       for name in models},
        }
    else:
        result = load(args.model_file).rank(args.request, limit=args.limit, min_score=args.min_score)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
