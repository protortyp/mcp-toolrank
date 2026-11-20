"""Fit and evaluate a cosine cutoff against a live gateway catalog."""
import asyncio
import json

from .rank_onnx import calibrate, read_cases


def ready_catalog(gateway):
    if any(upstream.state != "ready" for upstream in gateway.upstreams.values()):
        raise ValueError("Wait until all configured upstreams are available.")
    return gateway.catalog_entries()


def read_labels(path, eligible):
    try:
        cases = read_cases(json.loads(path.read_text(encoding="utf-8")), eligible)
    except (OSError, ValueError):
        raise ValueError("Provide a JSON array of query and relevant fields with eligible tool IDs.") from None
    if any(len(query) > 8192 for query, _ in cases):
        raise ValueError("Label queries must contain at most 8192 characters.")
    return cases


async def evaluate_gateway(gateway, cases, limit, invocable_only):
    supported = unsupported = accepted_supported = accepted_unsupported = hits = complete = 0
    for query, relevant in cases:
        result = await gateway.search(query, limit=limit, invocable_only=invocable_only)
        if result["decision"] == "calibration_scope_mismatch":
            raise ValueError("The configured calibration report does not match this catalog or search scope.")
        found = {match["id"] for match in result["matches"]}
        if relevant:
            supported += 1
            accepted_supported += bool(found)
            hits += bool(relevant & found)
            complete += relevant.issubset(found)
        else:
            unsupported += 1
            accepted_unsupported += bool(found)
    accepted = accepted_supported + accepted_unsupported
    return {"limit": limit, "supported": supported, "unsupported": unsupported,
            "accepted_supported": accepted_supported, "accepted_unsupported": accepted_unsupported,
            "supported_hit_count": hits, "supported_complete_count": complete,
            "supported_coverage": accepted_supported / supported if supported else None,
            "unsupported_false_accept_rate": accepted_unsupported / unsupported if unsupported else None,
            "selective_hit_rate": hits / accepted if accepted else None}


async def run_labeled(gateway, args):
    if args.calibrate and gateway.config.ranking != "hybrid":
        raise ValueError("Calibration requires hybrid ranking.")
    scope = args.invocable_only
    async with gateway.running():
        entries = ready_catalog(gateway)
        eligible = {entry.id for entry in entries if not scope or entry.invocable}
        if not eligible:
            raise ValueError("This search scope has no eligible tools.")
        cases = read_labels(args.calibrate or args.evaluate, eligible)
        before = gateway.calibration_scope(entries, scope)
        if args.calibrate:
            def scores(queries):
                return gateway.index.max_cosines(entries, queries, invocable_only=scope)

            report = await asyncio.to_thread(calibrate, cases, score_requests=scores,
                                             max_false_accept=args.max_false_accept, confidence=args.confidence)
        else:
            report = await evaluate_gateway(gateway, cases, args.limit, scope)
        if before != gateway.calibration_scope(ready_catalog(gateway), scope):
            raise ValueError("Catalog changed during the labeled run. Retry with a stable catalog.")
    return {"schema_version": 1, "eligible_tools": len(eligible), "scope": before.model_dump(), **report}
