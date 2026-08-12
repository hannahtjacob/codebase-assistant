import unittest

import numpy as np

from chunker import CodeChunk
from semantic_search import cosine_similarity, embed_texts, search


class FakeEmbeddingModel:
    """Deterministic test embeddings; the production code uses SentenceTransformer."""

    vectors = {
        "password check": [1.0, 0.0],
        "authenticate user credentials": [0.9, 0.1],
        "calculate payment": [0.0, 1.0],
        "unrelated": [-1.0, 0.0],
    }

    def encode(self, sentences):
        return np.asarray([self.vectors[text] for text in sentences])


def make_chunk(path: str, content: str) -> CodeChunk:
    return CodeChunk(path, path, "Python", 1, 1, content)


class CosineSimilarityTests(unittest.TestCase):
    def test_calculates_similarity_without_a_library_helper(self) -> None:
        scores = cosine_similarity(
            np.array([1.0, 0.0]),
            np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        )

        np.testing.assert_allclose(scores, [1.0, 1 / np.sqrt(2), 0.0])

    def test_zero_vector_has_zero_similarity(self) -> None:
        scores = cosine_similarity(np.zeros(2), np.array([[1.0, 0.0]]))

        np.testing.assert_array_equal(scores, [0.0])


class SemanticSearchTests(unittest.TestCase):
    def test_embed_texts_returns_numpy_matrix(self) -> None:
        vectors = embed_texts(["password check", "calculate payment"], FakeEmbeddingModel())

        self.assertEqual(vectors.shape, (2, 2))
        self.assertEqual(vectors.dtype, np.float32)

    def test_ranks_all_chunks_by_cosine_similarity(self) -> None:
        chunks = [
            make_chunk("payment.py", "calculate payment"),
            make_chunk("auth.py", "authenticate user credentials"),
            make_chunk("misc.py", "unrelated"),
        ]

        results = search("password check", chunks, FakeEmbeddingModel())

        self.assertEqual(
            [result.chunk.file_path for result in results],
            ["auth.py", "payment.py", "misc.py"],
        )
        self.assertGreater(results[0].score, results[1].score)

    def test_honors_default_top_five_and_explicit_limit(self) -> None:
        chunks = [make_chunk(str(index), "authenticate user credentials") for index in range(7)]

        self.assertEqual(len(search("password check", chunks, FakeEmbeddingModel())), 5)
        self.assertEqual(len(search("password check", chunks, FakeEmbeddingModel(), limit=2)), 2)

    def test_empty_chunks_and_invalid_limit(self) -> None:
        self.assertEqual(search("password check", [], FakeEmbeddingModel()), [])
        with self.assertRaises(ValueError):
            search("password check", [], FakeEmbeddingModel(), limit=-1)


if __name__ == "__main__":
    unittest.main()
