from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from toolrank.core import load_model


@dataclass(frozen=True)
class Match:
    tool: dict[str, Any]
    cosine: float


class ToolSelector:
    """Cache embeddings for one MCP catalog and return ranked tool definitions."""

    def __init__(self, model, tools: Sequence[Mapping[str, Any]]):
        self.model = model
        catalog = [dict(tool) for tool in tools]
        names = [tool.get("name") for tool in catalog]
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("each MCP tool needs a nonempty name")
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique within the catalog")
        self._tools = deepcopy(sorted(catalog, key=lambda tool: tool["name"]))
        self._vectors = None
        if self._tools:
            self._vectors = model.encode_document(
                [f"{tool['name']}: {tool.get('description') or ''}" for tool in self._tools],
                normalize_embeddings=True,
            )

    @classmethod
    def from_checkpoint(cls, model_id, tools):
        return cls(load_model(model_id), tools)

    def rank(self, query: str, *, k: int = 5, min_score: float | None = None) -> list[Match]:
        if k < 0:
            raise ValueError("k must be nonnegative")
        if self._vectors is None or k == 0 or not query.strip():
            return []
        vector = self.model.encode_query(query, normalize_embeddings=True)
        scores = self._vectors @ vector
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            Match(deepcopy(self._tools[i]), float(scores[i]))
            for i in order
            if min_score is None or scores[i] >= min_score
        ]
