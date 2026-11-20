from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from examples.rank_onnx import read_cached_vectors, write_cached_vectors


class CatalogCacheTests(unittest.TestCase):
    def test_precision_and_key_are_checked(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "test.npz"
            vectors = np.array([[1, 2, 3]], dtype=np.float32)
            write_cached_vectors(path, (vectors, None))
            actual, scales = read_cached_vectors(path, 1, 3, "float32")
            np.testing.assert_array_equal(vectors, actual)
            self.assertIsNone(scales)
            self.assertIsNone(read_cached_vectors(path, 1, 3, "int8"))
            moved = path.with_stem("other")
            path.rename(moved)
            self.assertIsNone(read_cached_vectors(moved, 1, 3, "float32"))

    def test_compact_cache_roundtrip_and_corruption(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "test.npz"
            vectors = np.array([[1], [2], [3]], dtype=np.int8)
            scales = np.array([0.01], dtype=np.float32)
            write_cached_vectors(path, (vectors, scales))
            actual_vectors, actual_scales = read_cached_vectors(path, 1, 3, "int8")
            np.testing.assert_array_equal(vectors, actual_vectors)
            np.testing.assert_array_equal(scales, actual_scales)
            path.write_bytes(b"PK\x03\x04partial archive")
            self.assertIsNone(read_cached_vectors(path, 1, 3, "int8"))
