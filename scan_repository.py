"""Find useful source files in a cloned repository."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


MAX_FILE_SIZE = 1_000_000  # 1 MB

LANGUAGES = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".java": "Java",
    ".cpp": "C++",
    ".h": "C/C++ Header",
}

IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "node_modules",
    "venv",
    "dist",
    "build",
}

LOCK_FILE_NAMES = {
    "bun.lock",
    "bun.lockb",
    "composer.lock",
    "gemfile.lock",
    "package-lock.json",
    "pipfile.lock",
    "poetry.lock",
    "yarn.lock",
}


@dataclass(frozen=True)
class SourceFile:
    """A text source file and the information needed to index it."""

    path: str
    language: str
    content: str


def _ignored_directory(name: str) -> bool:
    lower_name = name.lower()
    return lower_name in IGNORED_DIRECTORIES or "pycache" in lower_name


def _is_lock_file(path: Path) -> bool:
    lower_name = path.name.lower()
    return lower_name in LOCK_FILE_NAMES or lower_name.endswith(".lock")


def _read_text_source(path: Path) -> str | None:
    """Read UTF-8 text, returning None for large, binary, or invalid files."""
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return None
        content = path.read_bytes()
    except OSError:
        return None

    # A NUL byte is a reliable, inexpensive signal that a file is binary.
    if b"\x00" in content:
        return None

    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan_repository(repo_path: str | Path) -> list[SourceFile]:
    """Return supported source files beneath *repo_path*.

    Paths in the result are POSIX-style and relative to the repository root so
    they remain stable if the local clone is moved.
    """
    root = Path(repo_path)
    if not root.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {root}")

    source_files: list[SourceFile] = []
    for current_dir, directory_names, file_names in os.walk(root):
        directory_names[:] = sorted(
            name for name in directory_names if not _ignored_directory(name)
        )

        current_path = Path(current_dir)
        for file_name in sorted(file_names):
            path = current_path / file_name
            language = LANGUAGES.get(path.suffix.lower())
            if language is None or _is_lock_file(path) or path.is_symlink():
                continue

            content = _read_text_source(path)
            if content is None:
                continue

            source_files.append(
                SourceFile(
                    path=path.relative_to(root).as_posix(),
                    language=language,
                    content=content,
                )
            )

    return source_files


def print_summary(source_files: list[SourceFile]) -> None:
    """Print a human-readable count grouped by language."""
    counts = Counter(source_file.language for source_file in source_files)
    print(f"Found {len(source_files)} source files")
    if counts:
        print()
        width = max(len(language) for language in counts)
        for language in LANGUAGES.values():
            if language in counts:
                print(f"{language + ':':<{width + 1}} {counts[language]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo_path",
        nargs="?",
        default="data/repos/requests",
        help="repository to scan (default: data/repos/requests)",
    )
    args = parser.parse_args()
    print_summary(scan_repository(args.repo_path))


if __name__ == "__main__":
    main()
