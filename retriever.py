"""Retrieval pipeline joining Chroma similarity with SQLite chunk data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from chunker import CodeChunk
from vector_store import ChromaCodeSearch, VectorSearchResult


class SearchEngine(Protocol):
    def search(
        self, repository_id: str, question: str, top_k: int = 5
    ) -> list[VectorSearchResult]: ...


@dataclass(frozen=True)
class RetrievedChunk:
    """A ranked chunk for callers that need its similarity score."""

    chunk: CodeChunk
    score: float


class Retriever:
    """Run the question-to-vector-to-metadata retrieval pipeline."""

    def __init__(self, search_engine: SearchEngine | None = None) -> None:
        self.search_engine = search_engine or ChromaCodeSearch()

    def retrieve(
        self, repo_id: str, question: str, top_k: int = 5
    ) -> list[CodeChunk]:
        """Return ranked chunks without coupling callers to score details."""
        return [
            result.chunk
            for result in self.search_engine.search(repo_id, question, top_k)
        ]

    def retrieve_with_scores(
        self, repo_id: str, question: str, top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Return ranked chunks and cosine scores for display or evaluation."""
        return [
            RetrievedChunk(chunk=result.chunk, score=result.score)
            for result in self.search_engine.search(repo_id, question, top_k)
        ]
