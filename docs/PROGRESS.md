# Progress

- Started clean at `ba340d3`, matching remote main. Initial checks passed 21 Python and four browser tests. CPU package, ONNX model, benchmark snapshots and prior hybrid results were local.
- Ran five comparisons of pinned checkpoints with shared frozen BM25 and TF-IDF weights. BGE and ToolRank hybrids tied within uncertainty on ToolRet and native MCP. Skip retraining.
- Added paginated CPU stdio discovery, refresh and reconnect, stable IDs, separate discovery and invocation allowlists, definition checks and incremental embedding caching.
- Passed 27 Python, nine gateway and package, and six browser tests. The gateway discovered five tools across two reference servers and invoked the clock.
- Ran 132 local Qwen trials. The best methods completed 16/22 each versus 12/22 with all tools. ToolRank hybrid cut schema tokens by 85%.
- Measured rejection across catalogs. Use local cutoffs. Chromium verified unverified-match labels, schema size, runtime and BM25.
- Verified OpenAPI ranking, both agent frameworks through a compatible local endpoint, offline checksums, and wheels on Python 3.12 and 3.13. External spend was $0.
- Live NWS calls succeeded. Routing and rejection missed holdout targets. Qwen missed coverage and CPU latency targets. [NWS pilot](NWS_PILOT.md) · [Verifier](NWS_CAPABILITY.md)
- BM25, BGE and ToolRank kept all required tools at five on mostly small native BFCL menus. ToolRank saved 7.5% of JSON. [BFCL](BFCL_SHORTLIST.md)
- Explicit NWS and Crossref reads verified invocation filtering and optional byte caps. ToolRank top ten cut JSON 90.4% with 17/20 complete shortlists across 93 tools. [Mixed](MIXED_CATALOG.md)
- 93-tool shortlists saved context. Batch results repeated (24/24), including the dependent workflow failure. [Agent pilot](MIXED_AGENT_EVAL.md)
- MCP Inspector found five tools across two reference servers from a clean CPU wheel. The clock worked and discovery-only Git returned `not_available`. [Gateway](GATEWAY.md#mcp-inspector)
- First-proposal dispatch raised full-catalog completion from 3/7 to 6/7. Compaction saved 22.2% of first-turn schema tokens and lost one completion, failing the gate. Keep full schemas. [Workflows](DEPENDENT_WORKFLOWS.md)
- Streamable HTTP supports per-server bearer tokens and allowlists. The wheel found public Cloudflare docs and called a loopback clock. Tests verified refresh, session expiry, token isolation, and no replay on timeout or redirect. [Gateway](GATEWAY.md#streamable-http-upstreams)
- BM25 started offline with an empty model cache. On ETOM, first search took 0.32 s versus 9.42 s with cold hybrid embeddings. Hybrid led nDCG@10, 92.70 versus 88.68. [Gateway ranking](GATEWAY_RANKING.md)
- LangChain and Pydantic AI discover tools each turn and discard extra proposals. Each passed two live trials verifying exact HTTP-to-stdio timestamp handoff and final answers. [Framework integrations](FRAMEWORK_INTEGRATIONS.md)
