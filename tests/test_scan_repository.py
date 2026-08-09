import tempfile
import unittest
from pathlib import Path

from scan_repository import MAX_FILE_SIZE, SourceFile, scan_repository


class ScanRepositoryTests(unittest.TestCase):
    def test_finds_supported_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {
                "app.py": "Python",
                "src/app.js": "JavaScript",
                "src/app.ts": "TypeScript",
                "Example.java": "Java",
                "native/main.cpp": "C++",
                "native/main.h": "C/C++ Header",
            }
            for relative_path in expected:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"source for {relative_path}", encoding="utf-8")

            result = scan_repository(root)

            self.assertEqual(
                [(file.path, file.language) for file in result],
                sorted(expected.items()),
            )
            self.assertTrue(all(isinstance(file, SourceFile) for file in result))
            self.assertTrue(all(file.content.startswith("source") for file in result))

    def test_ignores_junk_directories_and_unsupported_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ignored_paths = [
                ".git/hook.py",
                "node_modules/library.js",
                "venv/module.py",
                ".venv/module.py",
                "__pycache__/cached.py",
                "custom_pycache_data/cached.py",
                "dist/bundle.js",
                "build/generated.java",
                "image.png",
                "README.md",
            ]
            for relative_path in ignored_paths:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("ignored", encoding="utf-8")

            self.assertEqual(scan_repository(root), [])

    def test_skips_binary_large_and_lock_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "binary.py").write_bytes(b"text\x00binary")
            (root / "invalid.js").write_bytes(b"\xff\xfe")
            (root / "generated.ts").write_bytes(b"x" * (MAX_FILE_SIZE + 1))
            (root / "example.lock").write_text("locked", encoding="utf-8")
            (root / "valid.py").write_text("print('valid')", encoding="utf-8")

            result = scan_repository(root)

            self.assertEqual([file.path for file in result], ["valid.py"])

    def test_rejects_a_missing_repository(self) -> None:
        with self.assertRaises(NotADirectoryError):
            scan_repository("this/path/does/not/exist")


if __name__ == "__main__":
    unittest.main()
