import json
import math
from pathlib import Path

MODELS = Path(".work/models")
BGE = "BAAI/bge-small-en-v1.5"
SCOUT = "nuhmanpk/toolscout-bge-small"
MICRO = "TaylorAI/bge-micro-v2"
REX = "Lux1997/Tool-Embed-0.6B"
PREFIX = "Represent this sentence for searching relevant passages: "


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_model(model_id, device="cpu", model_kwargs=None):
    from sentence_transformers import SentenceTransformer

    local = MODELS / model_id
    model = SentenceTransformer(
        str(local) if local.is_dir() else model_id,
        device=device,
        model_kwargs=model_kwargs,
        local_files_only=True,
    )
    if model_id in (BGE, SCOUT):
        model.prompts["query"] = PREFIX
    return model


def metrics(ranked, gold):
    positive = {key: value for key, value in gold.items() if value > 0}
    if not positive:
        raise ValueError("retrieval metrics require a positive judgment")
    result = {
        f"recall@{k}": sum(key in positive for key in ranked[:k]) / len(positive)
        for k in (1, 5, 10)
    }
    # trec_eval ndcg_cut uses linear graded gains.
    def dcg(grades):
        return sum(grade / math.log2(i + 2) for i, grade in enumerate(grades))
    ideal = sorted(positive.values(), reverse=True)[:10]
    result["ndcg@10"] = dcg([positive.get(key, 0) for key in ranked[:10]]) / dcg(ideal)
    result["complete@10"] = float(positive.keys() <= set(ranked[:10]))
    return result
