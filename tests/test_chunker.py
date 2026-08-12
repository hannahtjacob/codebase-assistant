import unittest

from chunker import CodeChunk, chunk_file, inspect_python_ast
from scan_repository import SourceFile


def make_source(line_count: int) -> SourceFile:
    content = "".join(f"line {number}\n" for number in range(1, line_count + 1))
    return SourceFile(path="src/example.py", language="Python", content=content)


class ChunkFileTests(unittest.TestCase):
    def test_creates_overlapping_windows(self) -> None:
        chunks = chunk_file(make_source(130))

        self.assertEqual(
            [(chunk.start_line, chunk.end_line) for chunk in chunks],
            [(1, 50), (41, 90), (81, 130)],
        )
        self.assertEqual(
            chunks[0].content.splitlines()[-10:],
            chunks[1].content.splitlines()[:10],
        )
        self.assertEqual(
            chunks[1].content.splitlines()[-10:],
            chunks[2].content.splitlines()[:10],
        )

    def test_short_file_produces_one_chunk(self) -> None:
        source = SourceFile("small.js", "JavaScript", "first\nsecond")

        result = chunk_file(source)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].start_line, 1)
        self.assertEqual(result[0].end_line, 2)
        self.assertEqual(result[0].content, source.content)
        self.assertIsInstance(result[0], CodeChunk)

    def test_empty_file_produces_no_chunks(self) -> None:
        source = SourceFile("empty.ts", "TypeScript", "")

        self.assertEqual(chunk_file(source), [])

    def test_ids_are_stable_and_content_sensitive(self) -> None:
        source = make_source(2)
        changed = SourceFile(source.path, source.language, "changed\ncontent\n")

        first_id = chunk_file(source)[0].id

        self.assertEqual(first_id, chunk_file(source)[0].id)
        self.assertNotEqual(first_id, chunk_file(changed)[0].id)

    def test_rejects_invalid_window_settings(self) -> None:
        source = make_source(10)
        invalid_settings = [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)]

        for chunk_size, overlap in invalid_settings:
            with self.subTest(chunk_size=chunk_size, overlap=overlap):
                with self.assertRaises(ValueError):
                    chunk_file(source, chunk_size=chunk_size, overlap=overlap)


class PythonAstChunkTests(unittest.TestCase):
    def test_chunks_complete_functions_and_classes(self) -> None:
        source = SourceFile(
            "src/auth.py",
            "Python",
            "import os\n\n"
            "@audit\n"
            "def authenticate_user(username, password):\n"
            "    if password:\n"
            "        return True\n"
            "    return False\n\n"
            "class Session:\n"
            "    def close(self):\n"
            "        pass\n",
        )

        chunks = chunk_file(source)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(
            (chunks[0].symbol_name, chunks[0].symbol_type),
            ("authenticate_user", "function"),
        )
        self.assertEqual((chunks[0].start_line, chunks[0].end_line), (3, 7))
        self.assertTrue(chunks[0].content.startswith("@audit\n"))
        self.assertIn("return False", chunks[0].content)
        self.assertEqual(
            (chunks[1].symbol_name, chunks[1].symbol_type),
            ("Session", "class"),
        )
        self.assertEqual((chunks[1].start_line, chunks[1].end_line), (9, 11))
        self.assertIn("def close", chunks[1].content)

    def test_async_function_is_a_complete_function_chunk(self) -> None:
        source = SourceFile(
            "worker.py",
            "Python",
            "async def fetch_data():\n    await client.get()\n",
        )

        chunk = chunk_file(source)[0]

        self.assertEqual(chunk.symbol_name, "fetch_data")
        self.assertEqual(chunk.symbol_type, "function")
        self.assertEqual((chunk.start_line, chunk.end_line), (1, 2))

    def test_valid_python_without_definitions_becomes_module_chunk(self) -> None:
        source = SourceFile("settings.py", "Python", "import os\nDEBUG = True\n")

        chunk = chunk_file(source)[0]

        self.assertEqual(chunk.symbol_name, "<module>")
        self.assertEqual(chunk.symbol_type, "module")
        self.assertEqual(chunk.content, source.content)

    def test_ast_experiment_exposes_important_node_types(self) -> None:
        dump = inspect_python_ast(
            "from api import client\n\n"
            "def login():\n"
            "    return client.auth.check()\n"
        )

        for node_type in (
            "Module",
            "FunctionDef",
            "ImportFrom",
            "Call",
            "Name",
            "Attribute",
        ):
            self.assertIn(node_type, dump)


if __name__ == "__main__":
    unittest.main()
