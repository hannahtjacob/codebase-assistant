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
        self.existing_ids = []
        self.existing_documents = []
        self.get_call = None
        self.delete_call = None

    def upsert(self, **kwargs):
        self.upsert_call = kwargs

    def query(self, **kwargs):
        self.query_call = kwargs
        return {
            "ids": [[deterministic_chunk_id("requests", make_chunk())]],
            "distances": [[0.125]],
        }

    def get(self, **kwargs):
        self.get_call = kwargs
        return {
            "ids": self.existing_ids,
            "documents": self.existing_documents,
        }

    def delete(self, **kwargs):
        self.delete_call = kwargs


class FakeClient:
    def __init__(self):
        self.collection = FakeCollection()
        self.collection_call = None

    def get_or_create_collection(self, **kwargs):
        self.collection_call = kwargs
        return self.collection


class FakeMetadataStore:
    def __init__(self):
        self.save_call = None
        self.query_call = None

    def save_index(self, **kwargs):
        self.save_call = kwargs

    def get_chunks(self, record_ids):
        return {record_ids[0]: make_chunk(chunk_id=record_ids[0])} if record_ids else {}

    def record_query(self, repository_id, question, top_k):
        self.query_call = (repository_id, question, top_k)


def make_chunk(chunk_id="auth-id", content="authenticate credentials"):
    return CodeChunk(chunk_id, "src/auth.py", "Python", 10, 20, content)


class ChromaCodeSearchTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.metadata = FakeMetadataStore()
        self.store = ChromaCodeSearch(
            client=self.client,
            model=FakeEmbeddingModel(),
            metadata_store=self.metadata,
        )

    def test_creates_a_cosine_collection_without_a_chroma_embedder(self):
        self.assertEqual(
            self.client.collection_call,
            {
                "name": "code_chunks",
                "embedding_function": None,
                "configuration": {"hnsw": {"space": "cosine"}},
            },
        )

    def test_index_splits_vectors_from_structured_metadata(self):
        count = self.store.index_chunks([make_chunk()], "requests")

        call = self.client.collection.upsert_call
        self.assertEqual(count, 1)
        self.assertNotIn("documents", call)
        self.assertEqual(call["embeddings"], [[0.8999999761581421, 0.10000000149011612]])
        self.assertEqual(call["metadatas"], [{"repository_id": "requests"}])
        self.assertEqual(len(call["ids"]), 1)
        self.assertEqual(self.metadata.save_call["chunks"], [make_chunk()])
        self.assertEqual(self.metadata.save_call["record_ids"], call["ids"])

    def test_repository_is_part_of_record_id(self):
        chunk = make_chunk()
        self.store.index_chunks([chunk], "one")
        first_id = self.client.collection.upsert_call["ids"][0]
        self.store.index_chunks([chunk], "two")
        second_id = self.client.collection.upsert_call["ids"][0]

        self.assertNotEqual(first_id, second_id)

    def test_index_removes_stale_chunks_for_the_same_repository(self):
        current_id = deterministic_chunk_id("requests", make_chunk())
        self.client.collection.existing_ids = [current_id, "old-window-id"]

        self.store.index_chunks([make_chunk()], "requests")

        self.assertEqual(self.client.collection.get_call, {
            "where": {"repository_id": "requests"},
            "include": ["documents"],
        })
        self.assertEqual(
            self.client.collection.delete_call,
            {"ids": ["old-window-id"]},
        )

    def test_index_removes_legacy_documents_from_chroma(self):
        current_id = deterministic_chunk_id("requests", make_chunk())
        self.client.collection.existing_ids = [current_id]
        self.client.collection.existing_documents = ["duplicated content"]

        self.store.index_chunks([make_chunk()], "requests")

        self.assertEqual(
            self.client.collection.delete_call,
            {"ids": [current_id]},
        )

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
            "include": ["distances"],
        })
        self.assertEqual(results[0].chunk.file_path, make_chunk().file_path)
        self.assertEqual(results[0].repository_id, "requests")
        self.assertAlmostEqual(results[0].score, 0.875)
        self.assertEqual(
            self.metadata.query_call,
            ("requests", "password check", 5),
        )

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
