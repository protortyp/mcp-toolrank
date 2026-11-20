import hashlib
from pathlib import Path
import re
import unicodedata

from sklearn.feature_extraction.text import HashingVectorizer

from toolrank.benchmark import local_corpus, toolret
from toolrank.core import write_json


def normalized(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def overlap(train, external):
    left = [normalized(text) for text in train]
    right = [normalized(text) for text in external]
    vectorizer = HashingVectorizer(analyzer="char", ngram_range=(4, 4), n_features=2**20, alternate_sign=False)
    maxima = (vectorizer.transform(left) @ vectorizer.transform(right).T).max(axis=1).toarray().ravel()
    return {
        "exact_distinct_strings": len(set(left) & set(right)),
        "train_strings_with_max_char4_cosine_ge_0_8": int((maxima >= 0.8).sum()),
        "max_char4_cosine": float(maxima.max()),
    }


def run():
    splits = {split: local_corpus(split, "plain") for split in ("train", "dev", "test")}
    train_tools, train_queries = splits["train"]
    report = {
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path("data").glob("*.json"))},
    }
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        a_tools, a_queries = splits[a]
        b_tools, b_queries = splits[b]
        counts = {
            "families": len({row[2] for row in a_queries} & {row[2] for row in b_queries}),
            "tool_ids": len(a_tools.keys() & b_tools.keys()),
            "tool_names": len({key.split("/")[1] for key in a_tools} & {key.split("/")[1] for key in b_tools}),
            "normalized_queries": len({normalized(row[1]) for row in a_queries} & {normalized(row[1]) for row in b_queries}),
        }
        if any(counts.values()):
            raise ValueError(f"split overlap {a}/{b}: {counts}")
        report[f"{a}_vs_{b}"] = counts
    public_tools, public_queries = toolret()
    report["train_vs_toolret"] = {
        "queries": overlap([row[1] for row in train_queries], [row[1] for row in public_queries]),
        "tool_text": overlap(list(train_tools.values()), list(public_tools.values())),
    }
    write_json("results/audit.json", report)


if __name__ == "__main__":
    run()
