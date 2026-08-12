"""Split Python by AST symbols and other source files by line windows."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

from scan_repository import SourceFile


@dataclass(frozen=True)
class CodeChunk:
    """A searchable section of a source file."""

    id: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    content: str
    symbol_name: str | None = None
    symbol_type: str | None = None


def _chunk_id(
    source_file: SourceFile, start_line: int, end_line: int, content: str
) -> str:
    """Create a stable ID that changes if the chunk's contents change."""
    identity = (
        f"{source_file.path}\0{source_file.language}\0"
        f"{start_line}\0{end_line}\0{content}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _line_chunks(
    source_file: SourceFile, chunk_size: int = 50, overlap: int = 10
) -> list[CodeChunk]:
    """Split a non-Python file into overlapping windows of lines."""
    lines = source_file.content.splitlines(keepends=True)
    if not lines:
        return []

    chunks: list[CodeChunk] = []
    step = chunk_size - overlap

    for start_index in range(0, len(lines), step):
        end_index = min(start_index + chunk_size, len(lines))
        content = "".join(lines[start_index:end_index])
        start_line = start_index + 1
        end_line = end_index
        chunks.append(
            CodeChunk(
                id=_chunk_id(source_file, start_line, end_line, content),
                file_path=source_file.path,
                language=source_file.language,
                start_line=start_line,
                end_line=end_line,
                content=content,
            )
        )

        if end_index == len(lines):
            break

    return chunks


def _python_chunks(source_file: SourceFile) -> list[CodeChunk] | None:
    """Create complete chunks for top-level Python functions and classes.

    ``None`` signals invalid Python so callers can fall back to line windows.
    Nested functions and methods remain inside their complete parent chunk.
    """
    try:
        tree = ast.parse(source_file.content)
    except (SyntaxError, ValueError):
        return None

    lines = source_file.content.splitlines(keepends=True)
    if not lines:
        return []

    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not definitions:
        content = "".join(lines)
        return [
            CodeChunk(
                id=_chunk_id(source_file, 1, len(lines), content),
                file_path=source_file.path,
                language=source_file.language,
                start_line=1,
                end_line=len(lines),
                content=content,
                symbol_name="<module>",
                symbol_type="module",
            )
        ]

    chunks: list[CodeChunk] = []
    for node in definitions:
        decorator_lines = [decorator.lineno for decorator in node.decorator_list]
        start_line = min([node.lineno, *decorator_lines])
        end_line = node.end_lineno
        if end_line is None:  # Defensive: end_lineno exists on supported Python.
            continue
        content = "".join(lines[start_line - 1 : end_line])
        symbol_type = "class" if isinstance(node, ast.ClassDef) else "function"
        chunks.append(
            CodeChunk(
                id=_chunk_id(source_file, start_line, end_line, content),
                file_path=source_file.path,
                language=source_file.language,
                start_line=start_line,
                end_line=end_line,
                content=content,
                symbol_name=node.name,
                symbol_type=symbol_type,
            )
        )
    return chunks


def inspect_python_ast(code: str) -> str:
    """Return an indented AST dump for a small Python-learning experiment."""
    return ast.dump(ast.parse(code), indent=2)


def chunk_file(
    source_file: SourceFile, chunk_size: int = 50, overlap: int = 10
) -> list[CodeChunk]:
    """Chunk Python by symbols and other languages by overlapping lines.

    Line numbers are 1-based and inclusive. Syntactically invalid Python falls
    back to line windows so partially edited files remain indexable.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if source_file.language == "Python":
        chunks = _python_chunks(source_file)
        if chunks is not None:
            return chunks
    return _line_chunks(source_file, chunk_size, overlap)
