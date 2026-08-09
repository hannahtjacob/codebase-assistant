import unittest

from chunker import CodeChunk
from keyword_search import search, tokenize


def make_chunk(path: str, content: str, start_line: int = 1) -> CodeChunk:
    line_count = max(1, len(content.splitlines()))
    return CodeChunk(
        id=f"{path}:{start_line}",
        file_path=path,
        language="Python",
        start_line=start_line,
        end_line=start_line + line_count - 1,
        content=content,
    )


class TokenizeTests(unittest.TestCase):
    def test_removes_question_words_and_normalizes_terms(self) -> None:
        self.assertEqual(
            tokenize("Where is the authentication token validated?"),
            ["authentication", "token", "validated"],
        )

    def test_splits_common_identifier_styles(self) -> None:
        self.assertEqual(
            tokenize("authenticate_user validateAccessToken HTTPResponse"),
            [
                "authenticate",
                "user",
                "validate",
                "access",
                "token",
                "http",
                "response",
            ],
        )


class SearchTests(unittest.TestCase):
    def test_ranks_by_keyword_frequency(self) -> None:
        chunks = [
            make_chunk("one.py", "authentication token"),
            make_chunk("two.py", "authentication token token validated"),
            make_chunk("three.py", "unrelated code"),
        ]

        results = search("Where is the authentication token validated?", chunks)

        self.assertEqual([result.chunk.file_path for result in results], ["two.py", "one.py"])
        self.assertEqual([result.score for result in results], [4, 2])
        self.assertEqual(
            results[0].matched_terms,
            ("authentication", "token", "validated"),
        )

    def test_exposes_synonym_failure(self) -> None:
        chunk = make_chunk("auth.py", "def authenticate_user(username, password): pass")

        results = search("Where does the app verify user credentials?", [chunk])

        # It sees only the generic word "user"; it cannot connect "verify
        # credentials" with "authenticate" without semantic understanding.
        self.assertEqual(results[0].score, 1)
        self.assertEqual(results[0].matched_terms, ("user",))

    def test_honors_limit_and_has_stable_tie_breaking(self) -> None:
        chunks = [make_chunk("b.py", "token"), make_chunk("a.py", "token")]

        results = search("token", chunks, limit=1)

        self.assertEqual([result.chunk.file_path for result in results], ["a.py"])

    def test_empty_or_stop_word_query_has_no_results(self) -> None:
        chunk = make_chunk("app.py", "the application")

        self.assertEqual(search("", [chunk]), [])
        self.assertEqual(search("where is it", [chunk]), [])

    def test_rejects_negative_limit(self) -> None:
        with self.assertRaises(ValueError):
            search("token", [], limit=-1)


if __name__ == "__main__":
    unittest.main()
