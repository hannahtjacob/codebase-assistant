import unittest

from chunker import CodeChunk, chunk_file
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


if __name__ == "__main__":
    unittest.main()
