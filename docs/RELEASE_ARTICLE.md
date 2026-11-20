# What improved MCP tool retrieval

We built a 33M-parameter retriever to shorten agent prompts. With matched hybrid scoring, fine-tuning added 0.70 nDCG@10 points on compact ETOM. Native MCP JSON and ToolRet intervals included zero. Lexical scoring raised BGE Small from 28.81 to 37.10 on ToolRet. [Matched experiments](BASELINE_COMPARISON.md)

The gateway discovers stdio and Streamable HTTP tools, preserves schemas, and checks permissions and revisions before calls. On 2,375 ETOM tools, model-free BM25 reached its first result in 0.32 seconds at 126 MiB peak RSS. Hybrid with uncached embeddings took 9.42 seconds at 287 MiB and scored 92.70 nDCG@10 versus BM25’s 88.68. [Gateway profiles](GATEWAY_RANKING.md)

With local Qwen 7B and 93 tools, a ten-tool shortlist saved 93.4% of first-turn schema tokens. Shortlisting completed five of seven dependent workflows. The full catalog completed six. Both ran one proposal per turn and replanned after each result. [Agent evaluation](DEPENDENT_WORKFLOWS.md)

Tiny catalogs may fit directly. NWS cosine rejection missed coverage and false-accept targets. The preview serves a trusted local client and requires explicit calls. [Calibration](GATEWAY_CALIBRATION.md) · [Try the preview](RELEASE_PREVIEW.md)
