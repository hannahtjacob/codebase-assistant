import unittest

import numpy as np

from chunker import CodeChunk
from vector_store import ChromaCodeSearch, deterministic_chunk_id


class FakeEmbeddingModel:
    vectors = {
        "password check": [1.0, 0.0],
        "authenticate credentials": [0.9, 0.1],
        "calculate payment": [0.0, 1.0],
    }

    def encode(self, sentences):
        return np.asarray([self.vectors[text] for text in sentences])


class FakeCollection:
    def __init__(self):
        self.upsert_call = None
        self.query_call = None

    def upsert(self, **kwargs):
        self.upsert_call = kwargs

    def query(self, **kwargs):
        self.query_call = kwargs
        return {
            "documents": [["authenticate credentials"]],
            "metadatas": [[{
                "chunk_id": "auth-id",
                "file_path": "src/auth.py",
                "language": "Python",
                "start_line": 10,
                "end_line": 20,
                "repository_id": "requests",
            }]],
            "distances": [[0.125]],
        }


class FakeClient:
    def __init__(self):
        self.collection = FakeCollection()
        self.collection_call = None

    def get_or_create_collection(self, **kwargs):
        self.collection_call = kwargs
        return self.collection


def make_chunk(chunk_id="auth-id", content="authenticate credentials"):
    return CodeChunk(chunk_id, "src/auth.py", "Python", 10, 20, content)


class ChromaCodeSearchTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.store = ChromaCodeSearch(client=self.client, model=FakeEmbeddingModel())

    def test_creates_a_cosine_collection_without_a_chroma_embedder(self):
        self.assertEqual(
            self.client.collection_call,
            {
                "name": "code_chunks",
                "embedding_function": None,
                "configuration": {"hnsw": {"space": "cosine"}},
            },
        )

    def test_index_chunks_stores_embeddings_documents_and_metadata(self):
        count = self.store.index_chunks([make_chunk()], "requests")

        call = self.client.collection.upsert_call
        self.assertEqual(count, 1)
        self.assertEqual(call["documents"], ["authenticate credentials"])
        self.assertEqual(call["embeddings"], [[0.8999999761581421, 0.10000000149011612]])
        self.assertEqual(call["metadatas"][0], {
            "chunk_id": "auth-id",
            "file_path": "src/auth.py",
            "language": "Python",
            "start_line": 10,
            "end_line": 20,
            "repository_id": "requests",
        })
        self.assertEqual(len(call["ids"]), 1)

    def test_repository_is_part_of_record_id(self):
        chunk = make_chunk()
        self.store.index_chunks([chunk], "one")
        first_id = self.client.collection.upsert_call["ids"][0]
        self.store.index_chunks([chunk], "two")
        second_id = self.client.collection.upsert_call["ids"][0]

        self.assertNotEqual(first_id, second_id)

    def test_record_id_is_deterministic_across_reindexing(self):
        chunk = make_chunk()

        self.store.index_chunks([chunk], "requests")
        first_id = self.client.collection.upsert_call["ids"][0]
        self.store.index_chunks([chunk], "requests")
        second_id = self.client.collection.upsert_call["ids"][0]

        self.assertEqual(first_id, second_id)
        self.assertEqual(first_id, deterministic_chunk_id("requests", chunk))
        self.assertEqual(len(first_id), 64)

    def test_search_filters_repository_and_rebuilds_chunk(self):
        results = self.store.search("requests", "password check")

        self.assertEqual(self.client.collection.query_call, {
            "query_embeddings": [[1.0, 0.0]],
            "where": {"repository_id": "requests"},
            "n_results": 5,
            "include": ["documents", "metadatas", "distances"],
        })
        self.assertEqual(results[0].chunk, make_chunk())
        self.assertEqual(results[0].repository_id, "requests")
        self.assertAlmostEqual(results[0].score, 0.875)

    def test_validates_repository_and_limits(self):
        with self.assertRaises(ValueError):
            self.store.index_chunks([make_chunk()], "")
        with self.assertRaises(ValueError):
            self.store.search("", "question")
        with self.assertRaises(ValueError):
            self.store.search("requests", "question", -1)
        self.assertEqual(self.store.search("requests", "question", 0), [])


if __name__ == "__main__":
    unittest.main()
