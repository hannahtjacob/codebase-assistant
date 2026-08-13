import unittest

from chunker import CodeChunk
from rag import RAGService, build_prompt, has_valid_citation


def make_chunk():
    return CodeChunk(
        "id",
        "src/auth.py",
        "Python",
        35,
        38,
        "def authenticate(user):\n    return check(user.password)\n",
        "authenticate",
        "function",
    )


class FakeRetriever:
    def __init__(self, chunks=None):
        self.chunks = [make_chunk()] if chunks is None else chunks
        self.call = None

    def retrieve(self, repo_id, question, top_k=5):
        self.call = (repo_id, question, top_k)
        return self.chunks


class FakeProvider:
    def __init__(self, answers=None):
        self.prompt = None
        self.prompts = []
        self.answers = answers or ["Authentication occurs in `src/auth.py:35-38`."]

    def generate(self, prompt):
        self.prompt = prompt
        self.prompts.append(prompt)
        return self.answers[min(len(self.prompts) - 1, len(self.answers) - 1)]


class PromptTests(unittest.TestCase):
    def test_prompt_contains_question_code_and_exact_citation(self):
        prompt = build_prompt("Where is authentication?", [make_chunk()])

        self.assertIn("Use ONLY the provided source code", prompt)
        self.assertIn("TASK (answer this question only):\nWhere is authentication?", prompt)
        self.assertIn("Citation: `src/auth.py:35-38`", prompt)
        self.assertIn("def authenticate(user):", prompt)
        self.assertIn("Every claim", prompt)

    def test_recognizes_only_exact_retrieved_citations(self):
        self.assertTrue(
            has_valid_citation("See `src/auth.py:35-38`.", [make_chunk()])
        )
        self.assertFalse(
            has_valid_citation("See `other.py:1-2`.", [make_chunk()])
        )


class RAGServiceTests(unittest.TestCase):
    def test_retrieves_then_generates_a_grounded_answer(self):
        retriever = FakeRetriever()
        provider = FakeProvider()
        service = RAGService(retriever, provider)

        result = service.answer("requests", "Where is authentication?", 3)

        self.assertEqual(retriever.call, ("requests", "Where is authentication?", 3))
        self.assertIn("src/auth.py:35-38", result.answer)
        self.assertEqual(result.sources, (make_chunk(),))
        self.assertIn("Source 1:", provider.prompt)

    def test_retries_an_answer_without_a_valid_citation(self):
        provider = FakeProvider([
            "Authentication happens in the auth module.",
            "Authentication happens in `src/auth.py:35-38`.",
        ])

        result = RAGService(FakeRetriever(), provider).answer("requests", "Where?")

        self.assertEqual(len(provider.prompts), 2)
        self.assertIn("prior response was rejected", provider.prompts[1])
        self.assertIn("`src/auth.py:35-38`", result.answer)

    def test_rejects_repeatedly_ungrounded_model_output(self):
        provider = FakeProvider(["No citation.", "Still no citation."])

        result = RAGService(FakeRetriever(), provider).answer("requests", "Where?")

        self.assertIn("did not produce a sufficiently grounded answer", result.answer)
        self.assertIn("`src/auth.py:35-38`", result.answer)

    def test_does_not_call_llm_without_retrieved_context(self):
        retriever = FakeRetriever([])
        provider = FakeProvider()

        result = RAGService(retriever, provider).answer("requests", "Unknown?")

        self.assertIn("could not find", result.answer)
        self.assertIsNone(provider.prompt)
        self.assertEqual(result.sources, ())

    def test_validates_inputs(self):
        service = RAGService(FakeRetriever(), FakeProvider())
        for repo_id, question, top_k in (("", "q", 5), ("repo", "", 5), ("repo", "q", -1)):
            with self.subTest(repo_id=repo_id, question=question, top_k=top_k):
                with self.assertRaises(ValueError):
                    service.answer(repo_id, question, top_k)


if __name__ == "__main__":
    unittest.main()
