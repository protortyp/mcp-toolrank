import json
import re

import numpy as np


_IDENTIFIER = re.compile(r"^[A-Za-z_][\w.-]*$")


def lexical_text(text: str) -> str:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return " ".join(re.findall(r"[a-z]+|[0-9]+", separated.lower()))


def repeat_name(text: str) -> str:
    if text.startswith("{"):
        try:
            name = json.loads(text).get("name")
        except (json.JSONDecodeError, AttributeError):
            name = None
    else:
        prefix, separator, _ = text.partition(":")
        name = prefix if separator and _IDENTIFIER.fullmatch(prefix) else None
    return f"{name} {name} {text}" if isinstance(name, str) else text


class Bm25Index:
    """BM25 over descriptions with extra weight on tool names."""

    def __init__(self, documents):
        from sklearn.feature_extraction.text import CountVectorizer

        self.count = len(documents)
        vectorizer = CountVectorizer(preprocessor=lexical_text, dtype=np.float32)
        weighted = [repeat_name(text) for text in documents]
        analyze = vectorizer.build_analyzer()
        self.postings = None
        if not any(analyze(text) for text in weighted):
            return
        postings = vectorizer.fit_transform(weighted).tocsc()
        frequencies = np.diff(postings.indptr)
        idf = np.log1p((self.count - frequencies + .5) / (frequencies + .5))
        lengths = np.asarray(postings.sum(axis=1)).ravel()
        length_penalty = 1.2 * (.25 + .75 * lengths / lengths.mean())
        postings.data *= 2.2 / (postings.data + length_penalty[postings.indices])
        postings.data *= np.repeat(idf, frequencies)
        self.vectorizer = vectorizer
        self.postings = postings.T.tocsr()

    def scores(self, queries):
        if self.postings is None:
            return np.zeros((len(queries), self.count), dtype=np.float32)
        terms = self.vectorizer.transform(queries).sign()
        scores = (terms @ self.postings).toarray()
        maximum = np.maximum(scores.max(axis=1, keepdims=True), 1e-9)
        return scores / maximum
