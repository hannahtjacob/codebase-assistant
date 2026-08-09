"""Split source files into overlapping, line-based chunks."""

from __future__ import annotations

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


def _chunk_id(
    source_file: SourceFile, start_line: int, end_line: int, content: str
) -> str:
    """Create a stable ID that changes if the chunk's contents change."""
    identity = (
        f"{source_file.path}\0{source_file.language}\0"
        f"{start_line}\0{end_line}\0{content}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def chunk_file(
    source_file: SourceFile, chunk_size: int = 50, overlap: int = 10
) -> list[CodeChunk]:
    """Split *source_file* into overlapping windows of lines.

    Line numbers are 1-based and inclusive. Empty files produce no chunks.
    ``overlap`` must be smaller than ``chunk_size`` so each window advances.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

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
