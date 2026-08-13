import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from chunker import CodeChunk
from metadata_store import Chunk, MetadataStore, QueryHistory, Repository, initialize_database


def make_chunk(content="def login():\n    return True\n"):
    return CodeChunk(
        "source-id",
        "src/auth.py",
        "Python",
        4,
        5,
        content,
        "login",
        "function",
    )


class RawSchemaTests(unittest.TestCase):
    def test_manual_schema_supports_exact_sql_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.db"
            initialize_database(path)
            with closing(sqlite3.connect(path)) as connection:
                table_names = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }

            self.assertTrue(
                {"repositories", "chunks", "query_history"}.issubset(table_names)
            )


class MetadataStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "metadata.db"
        self.store = MetadataStore(self.path)

    def tearDown(self):
        self.store.engine.dispose()
        self.temporary_directory.cleanup()

    def test_saves_repository_and_chunk_metadata(self):
        chunk = make_chunk()
        self.store.save_index(
            "requests", "https://github.com/psf/requests", "requests", "abc123",
            [chunk], ["record-id"],
        )

        with Session(self.store.engine) as session:
            repository = session.get(Repository, "requests")
            stored = session.get(Chunk, "record-id")
            self.assertEqual(repository.commit_hash, "abc123")
            self.assertEqual(stored.symbol_name, "login")
            self.assertEqual(
                stored.content_hash,
                hashlib.sha256(chunk.content.encode("utf-8")).hexdigest(),
            )

        self.assertEqual(self.store.get_chunks(["record-id"])["record-id"].content, chunk.content)

    def test_reindex_replaces_stale_chunk_rows(self):
        self.store.save_index("repo", "url", "name", "one", [make_chunk()], ["old"])
        self.store.save_index("repo", "url", "name", "two", [make_chunk("new")], ["new"])

        with Session(self.store.engine) as session:
            ids = list(session.scalars(select(Chunk.id)))
        self.assertEqual(ids, ["new"])

    def test_records_query_history(self):
        self.store.save_index("repo", "url", "name", "commit", [], [])
        self.store.record_query("repo", "Where is login checked?", 5)

        with Session(self.store.engine) as session:
            query = session.scalar(select(QueryHistory))
            self.assertEqual(query.repository_id, "repo")
            self.assertEqual(query.question, "Where is login checked?")
            self.assertEqual(query.top_k, 5)


if __name__ == "__main__":
    unittest.main()
