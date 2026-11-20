# Progress

- Started clean at `ba340d3`, matching remote main. Initial checks passed 21 Python and four browser tests. CPU package, ONNX model, benchmark snapshots and prior hybrid results were local.
- Ran five comparisons of pinned checkpoints with shared frozen BM25 and TF-IDF weights. BGE and ToolRank hybrids tied within uncertainty on ToolRet and native MCP. Skip retraining.
- Added paginated CPU stdio discovery, refresh and reconnect, stable IDs, separate discovery and invocation allowlists, definition checks and incremental embedding caching.
- Passed 27 Python, nine gateway and package, and six browser tests. The gateway discovered five tools across two reference servers and invoked the clock.
- Ran 132 local Qwen trials. The best methods completed 16/22 each versus 12/22 with all tools. ToolRank hybrid cut schema tokens by 85%.
- Measured rejection across catalogs. Use local cutoffs. Chromium verified unverified-match labels, schema size, runtime and BM25.
- Verified OpenAPI ranking, both agent frameworks through a compatible local endpoint, offline checksums, and wheels on Python 3.12 and 3.13. External spend was $0.
