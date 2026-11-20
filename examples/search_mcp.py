# /// script
# requires-python = ">=3.12"
# dependencies = ["mcp==1.30.0", "numpy==2.5.5", "onnxruntime==1.24.1", "tokenizers==0.22.2"]
# ///
"""Serve local ToolRank catalog search over MCP stdio."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel

from rank_onnx import OnnxToolRank, read_catalog


class Match(BaseModel):
    tool: dict[str, Any]
    score: float


class SearchResult(BaseModel):
    matches: list[Match]


def create_server(retriever: OnnxToolRank) -> FastMCP:
    mcp = FastMCP("ToolRank")

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                                          idempotentHint=True, openWorldHint=False))
    def search_tools(query: str, limit: int = 5, min_score: float | None = None) -> SearchResult:
        """Return ranked tool definitions for a natural-language request.

        The optional min_score filters by cosine similarity. An empty match list
        means no tool passed the threshold or the catalog is empty.
        """
        if not query.strip():
            raise ValueError("query must not be blank")
        if limit < 1:
            raise ValueError("limit must be positive")
        if min_score is not None and not math.isfinite(min_score):
            raise ValueError("min_score must be finite")
        rows = retriever.rank(query, limit=limit, min_score=min_score)
        return SearchResult(matches=[Match(**row) for row in rows])

    return mcp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("catalog", type=Path, help="an MCP tools/list result or a tool array")
    parser.add_argument("--model-file", default="embedding-int8.onnx")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--catalog-precision", choices=["float32", "int8"], default="float32")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        tools = read_catalog(json.loads(args.catalog.read_text()))
        retriever = OnnxToolRank(args.checkpoint, tools, threads=args.threads, model_file=args.model_file,
                                 catalog_precision=args.catalog_precision)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    create_server(retriever).run(transport="stdio")


if __name__ == "__main__":
    main()
