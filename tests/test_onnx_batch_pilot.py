import unittest

import numpy as np

from toolrank.onnx_batch_pilot import score_dataset


class FixedEncoder:
    prefix = "query: "

    def encode(self, texts):
        rows = np.zeros((len(texts), 384), dtype=np.float32)
        for row, text in zip(rows, texts, strict=True):
            row[0 if "station" in text else 1] = 1
        return rows


class BatchPilotTests(unittest.TestCase):
    def test_bucketed_index_maps_vectors_back_to_stable_tool_ids(self):
        docs = {"z": "track: Train station.", "a": "motion: Detect movement in a password-protected gallery."}
        cases = [("train station", "rail", {"z": 1}), ("detect movement", "gallery", {"a": 1})]
        result = score_dataset(FixedEncoder(), docs, cases, batch=1)
        self.assertEqual(result["same_top10_fraction"], {"dense": 1, "hybrid": 1})
        self.assertEqual(result["max_catalog_embedding_delta"], 0)
        for order in ("stable", "bucketed"):
            for ranker in ("dense", "hybrid"):
                self.assertEqual([row["ndcg@10"] for row in result["per_query"][order][ranker]], [1, 1])
