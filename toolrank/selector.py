from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from toolrank.core import load_model
from toolrank.lexical import Bm25Index, lexical_text


@dataclass(frozen=True)
class Match:
    tool: dict[str, Any]
    cosine: float


class ToolSelector:
    """Cache embeddings for one MCP catalog and return ranked tool definitions."""

    def __init__(self, model, tools: Sequence[Mapping[str, Any]], *, lexical_weight: float = 0.0,
                 bm25_weight: float = 0.0):
        if not np.isfinite(lexical_weight) or lexical_weight < 0:
            raise ValueError("lexical_weight must be finite and nonnegative")
        if not np.isfinite(bm25_weight) or bm25_weight < 0:
            raise ValueError("bm25_weight must be finite and nonnegative")
        self.model = model
        catalog = [dict(tool) for tool in tools]
        names = [tool.get("name") for tool in catalog]
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("each MCP tool needs a nonempty name")
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique within the catalog")
        self._tools = deepcopy(sorted(catalog, key=lambda tool: tool["name"]))
        self._vectors = None
        self._lexical_weight = lexical_weight
        self._bm25_weight = bm25_weight
        self._bm25_index = None
        self._vectorizer = None
        self._lexical_vectors = None
        if not self._tools:
            return
        descriptions = [f"{tool['name']}: {tool.get('description') or ''}" for tool in self._tools]
        self._vectors = model.encode_document(descriptions, normalize_embeddings=True)
        if bm25_weight:
            self._bm25_index = Bm25Index(descriptions)
        if lexical_weight:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(preprocessor=lexical_text, ngram_range=(1, 2))
            analyze = vectorizer.build_analyzer()
            if any(analyze(description) for description in descriptions):
                self._lexical_vectors = vectorizer.fit_transform(descriptions)
                self._vectorizer = vectorizer

    @classmethod
    def from_checkpoint(cls, model_id, tools, *, lexical_weight: float = 0.0, bm25_weight: float = 0.0):
        return cls(load_model(model_id), tools, lexical_weight=lexical_weight, bm25_weight=bm25_weight)

    def rank(self, query: str, *, k: int = 5, min_score: float | None = None) -> list[Match]:
        if k < 0:
            raise ValueError("k must be nonnegative")
        if self._vectors is None or k == 0 or not query.strip():
            return []
        vector = self.model.encode_query(query, normalize_embeddings=True)
        cosine = self._vectors @ vector
        scores = cosine
        if self._vectorizer is not None:
            query_terms = self._vectorizer.transform([query])
            similarities = (query_terms @ self._lexical_vectors.T).toarray()[0]
            scores = cosine + self._lexical_weight * similarities
        if self._bm25_index is not None:
            scores = scores + self._bm25_weight * self._bm25_index.scores([query])[0]
        order = np.argsort(-scores, kind="stable")
        if min_score is not None:
            order = order[cosine[order] >= min_score]
        return [
            Match(deepcopy(self._tools[i]), float(cosine[i]))
            for i in order[:k]
        ]
