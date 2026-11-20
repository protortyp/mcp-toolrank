# What improved MCP tool retrieval

Fine-tuning 33M-parameter BGE Small helped on compact descriptions. On native MCP JSON and ToolRet, it made little difference when both models used the same scoring. TF-IDF and BM25 raised BGE Small from 28.81 to 37.10 nDCG@10 on ToolRet. Fine-tuning added about a tenth of a point. [Reproduction and results](BASELINE_COMPARISON.md)

The gateway searches several local stdio catalogs, distinguishes duplicate names, preserves schemas, and checks allowlists and definition revisions before routing explicit calls. It uses the existing 34 MB CPU ONNX checkpoint. The browser shows what a five-tool shortlist would put in context. [Gateway](GATEWAY.md)

In a small agent test, ToolRank hybrid completed 16 of 22 trials with 85% fewer schema tokens than the all-tools arm, which completed 12. BGE dense also completed 16. Both hybrids struggled when available tools lacked permission or capability. [Agent evaluation](AGENT_EVALUATION.md)

For small, stable catalogs, consider passing schemas directly. For centralized authentication, choose a gateway built for that deployment. [Scope and limits](GATEWAY.md)
