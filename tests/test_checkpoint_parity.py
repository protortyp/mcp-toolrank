"""Optional parity check with pinned release model files already on disk."""
import os
from pathlib import Path
import unittest


class CheckpointParityTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("TOOLRANK_CHECKPOINT") and os.environ.get("TOOLRANK_INT8"),
                         "requires local release model snapshots")
    def test_pytorch_and_onnx_agree_on_compact_api_ranking(self):
        import numpy as np
        import torch

        from examples.rank_onnx import OnnxToolRank
        from toolrank.core import load_model, read_json

        torch.set_num_threads(4)
        checkpoint = Path(os.environ["TOOLRANK_CHECKPOINT"]).resolve()
        data = read_json(Path("data/onnx_int8_holdout.json"))
        catalog = sorted(data["tools"].values(), key=lambda tool: tool["name"])
        texts = [f"{row['name']}: {row['description']}" for row in catalog]
        questions = [row["query"] for row in data["queries"]]
        model = load_model(str(checkpoint))
        options = dict(normalize_embeddings=True, show_progress_bar=False)
        reference = model.encode_query(questions, **options) @ model.encode_document(texts, **options).T
        for folder, graph, similarity_limit, agreement_limit in (
            (checkpoint, "embedding.onnx", .001, .98),
            (Path(os.environ["TOOLRANK_INT8"]), "embedding-int8.onnx", .20, .80),
        ):
            with self.subTest(graph=graph):
                encoder = OnnxToolRank(folder, catalog, model_file=graph)
                candidate = np.stack([encoder.cosine_scores(query) for query in questions])
                mean_error = float(np.abs(candidate - reference).mean())
                agreement = float((candidate.argmax(axis=1) == reference.argmax(axis=1)).mean())
                print(graph, "mean absolute cosine error", mean_error, "top-one agreement", agreement)
                self.assertLess(mean_error, similarity_limit)
                self.assertGreaterEqual(agreement, agreement_limit)
