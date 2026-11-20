"""NumPy lexical scores for the standalone ONNX CLI and MCP server."""
from collections import Counter
from itertools import pairwise
import json
import re

import numpy as np


_WORDS = re.compile(r"[a-z]+|[0-9]+")
_CAMEL_CASE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_IDENTIFIER = re.compile(r"^[A-Za-z_][\w.-]*$")


def words(text):
    separated = _CAMEL_CASE.sub(" ", text)
    return [token for token in _WORDS.findall(separated.lower()) if len(token) >= 2]


def terms(tokens):
    return Counter([*tokens, *(" ".join(pair) for pair in pairwise(tokens))])


def weighted_tokens(document, tokens):
    if document.startswith("{"):
        try:
            name = json.loads(document).get("name")
        except (json.JSONDecodeError, AttributeError):
            name = None
    else:
        prefix, separator, _ = document.partition(":")
        name = prefix if separator and _IDENTIFIER.fullmatch(prefix) else None
    return [*(words(name) * 2), *tokens] if isinstance(name, str) else tokens


class Postings:
    """Store an inverted index in contiguous arrays."""

    def __init__(self, count_documents, dtype):
        frequencies = Counter()
        for counts in count_documents():
            frequencies.update(counts.keys())
        self.terms = {term: index for index, term in enumerate(frequencies)}
        self.frequencies = np.fromiter(frequencies.values(), dtype=np.intp)
        self.offsets = np.empty(len(self.terms) + 1, dtype=np.intp)
        self.offsets[0] = 0
        np.cumsum(self.frequencies, out=self.offsets[1:])
        self.rows = np.empty(self.offsets[-1], dtype=np.int32)
        self.weights = np.empty(self.offsets[-1], dtype=dtype)
        positions = self.offsets[:-1].copy()
        for row, frequencies in enumerate(count_documents()):
            indices = np.fromiter((self.terms[term] for term in frequencies), dtype=np.intp)
            destinations = positions[indices]
            self.rows[destinations] = row
            self.weights[destinations] = np.fromiter(frequencies.values(), dtype=dtype)
            positions[indices] += 1

    def entries(self, index):
        span = slice(self.offsets[index], self.offsets[index + 1])
        return self.rows[span], self.weights[span]


class CatalogLexicalIndex:
    def __init__(self, documents):
        self.size = len(documents)
        tokens = [words(text) for text in documents]
        self.tfidf = self._tfidf(tokens)
        self.bm25 = self._bm25(documents, tokens)

    def _tfidf(self, tokens):
        postings = Postings(lambda: map(terms, tokens), np.float64)
        postings.idf = 1 + np.log((self.size + 1) / (postings.frequencies + 1))
        postings.weights *= np.repeat(postings.idf, postings.frequencies)
        squares = np.bincount(postings.rows, weights=postings.weights ** 2, minlength=self.size)
        postings.weights /= np.sqrt(squares[postings.rows])
        return postings

    def _bm25(self, documents, tokens):
        weighted = [weighted_tokens(document, row) for document, row in zip(documents, tokens, strict=True)]
        lengths = np.array([len(row) for row in weighted], dtype=np.float32)
        postings = Postings(lambda: map(Counter, weighted), np.float32)
        if lengths.any():
            penalties = 1.2 * (.25 + .75 * lengths / lengths.mean())
            postings.weights *= 2.2 / (postings.weights + penalties[postings.rows])
            frequencies = postings.frequencies
            idf = np.log1p((self.size - frequencies + .5) / (frequencies + .5))
            postings.weights *= np.repeat(idf, frequencies)
        return postings

    def scores(self, request):
        tokens = words(request)
        weights = terms(tokens)
        tfidf = np.zeros(self.size, dtype=np.float64)
        norm_squared = 0.0
        for term, count in weights.items():
            if (index := self.tfidf.terms.get(term)) is not None:
                rows, values = self.tfidf.entries(index)
                weighted_query = count * self.tfidf.idf[index]
                norm_squared += weighted_query * weighted_query
                tfidf[rows] += values * weighted_query
        if norm_squared:
            tfidf /= np.sqrt(norm_squared)

        bm25 = np.zeros(self.size, dtype=np.float32)
        for token in dict.fromkeys(tokens):
            if (index := self.bm25.terms.get(token)) is not None:
                rows, values = self.bm25.entries(index)
                bm25[rows] += values
        bm25 /= max(bm25.max(initial=0), 1e-9)
        return tfidf, bm25

    def blend(self, request, cosine):
        tfidf, bm25 = self.scores(request)
        return cosine + .4 * tfidf + .15 * bm25
