import unittest

from chunker import CodeChunk
from retriever import RetrievedChunk, Retriever
from search import format_results
from vector_store import VectorSearchResult


def make_chunk(path, start_line, end_line, symbol_name=None):
    return CodeChunk(
        path,
        path,
        "Python",
        start_line,
        end_line,
        "code",
        symbol_name,
        "function" if symbol_name else None,
    )


class FakeSearchEngine:
    def __init__(self):
        self.call = None
        self.results = [
            VectorSearchResult(
                make_chunk("requests/sessions.py", 356, 450, "create_session"),
                "requests",
                0.81,
            ),
            VectorSearchResult(
                make_chunk("requests/api.py", 10, 60),
                "requests",
                0.73,
            ),
        ]

    def search(self, repository_id, question, top_k=5):
        self.call = (repository_id, question, top_k)
        return self.results[:top_k]


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeSearchEngine()
        self.retriever = Retriever(self.engine)

    def test_retrieve_returns_ranked_code_chunks(self):
        chunks = self.retriever.retrieve(
            "requests", "Where are HTTP sessions created?", 1
        )

        self.assertEqual(self.engine.call, (
            "requests", "Where are HTTP sessions created?", 1
        ))
        self.assertEqual(len(chunks), 1)
        self.assertIsInstance(chunks[0], CodeChunk)
        self.assertEqual(chunks[0].file_path, "requests/sessions.py")

    def test_scored_retrieval_keeps_ranking_scores(self):
        results = self.retriever.retrieve_with_scores(
            "requests", "Where are HTTP sessions created?", 2
        )

        self.assertEqual([result.score for result in results], [0.81, 0.73])

    def test_formats_cli_results(self):
        output = format_results([
            RetrievedChunk(self.engine.results[0].chunk, 0.81),
            RetrievedChunk(self.engine.results[1].chunk, 0.73),
        ])

        self.assertIn("1. requests/sessions.py:356-450", output)
        self.assertIn("score=0.8100  function: create_session", output)
        self.assertIn("2. requests/api.py:10-60", output)

    def test_formats_empty_results(self):
        self.assertEqual(format_results([]), "No matching chunks found.")


if __name__ == "__main__":
    unittest.main()
