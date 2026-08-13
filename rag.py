"""Retrieval-augmented answer generation over indexed source code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from chunker import CodeChunk
from llm_provider import LLMProvider
from retriever import Retriever


class ChunkRetriever(Protocol):
    def retrieve(
        self, repo_id: str, question: str, top_k: int = 5
    ) -> list[CodeChunk]: ...


def source_citation(chunk: CodeChunk) -> str:
    return f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"


def build_prompt(question: str, chunks: list[CodeChunk]) -> str:
    """Create a grounded prompt containing the question and retrieved code."""
    sources: list[str] = []
    for number, chunk in enumerate(chunks, start=1):
        citation = source_citation(chunk)
        sources.append(
            f"Source {number}:\n"
            f"Citation: `{citation}`\n"
            f"File: {chunk.file_path}\n"
            f"Lines: {chunk.start_line}-{chunk.end_line}\n"
            f"```{chunk.language.lower()}\n{chunk.content}\n```"
        )

    return (
        "You answer questions about a codebase.\n\n"
        "Use ONLY the provided source code. Do not use outside knowledge or "
        "invent missing behavior. If the sources are insufficient, say so. "
        "Every claim about the code must include an exact citation in backticks "
        "using one of the provided file:line-line citations. Do not summarize "
        "all sources; answer only the specific question in 1-3 sentences.\n\n"
        + "\n\n".join(sources)
        + "\n\nTASK (answer this question only):\n"
        + question
        + "\n\nAnswer in 1-3 sentences and cite every sentence with an exact "
        "backticked Citation value shown above."
    )


def has_valid_citation(answer: str, chunks: list[CodeChunk]) -> bool:
    """Check that an answer cites at least one exact retrieved source range."""
    return any(f"`{source_citation(chunk)}`" in answer for chunk in chunks)


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    sources: tuple[CodeChunk, ...]


class RAGService:
    """Retrieve source context, augment a prompt, and generate an answer."""

    def __init__(self, retriever: ChunkRetriever, provider: LLMProvider) -> None:
        self.retriever = retriever
        self.provider = provider

    def answer(
        self, repo_id: str, question: str, top_k: int = 5
    ) -> RAGAnswer:
        if not repo_id.strip():
            raise ValueError("repo_id cannot be empty")
        if not question.strip():
            raise ValueError("question cannot be empty")
        if top_k < 0:
            raise ValueError("top_k cannot be negative")

        chunks = self.retriever.retrieve(repo_id, question, top_k)
        if not chunks:
            return RAGAnswer(
                answer="I could not find relevant indexed source code to answer this question.",
                sources=(),
            )
        prompt = build_prompt(question, chunks)
        answer = self.provider.generate(prompt)
        if not has_valid_citation(answer, chunks):
            correction = (
                f"{prompt}\n\n"
                "CORRECTION: Return only a direct 1-3 sentence answer to the "
                "TASK. Your prior response was rejected. Every sentence must "
                "contain an exact backticked Citation value from above."
            )
            answer = self.provider.generate(correction)
        if not has_valid_citation(answer, chunks):
            citations = ", ".join(
                f"`{source_citation(chunk)}`" for chunk in chunks
            )
            answer = (
                "The model did not produce a sufficiently grounded answer. "
                f"Relevant retrieved sources: {citations}."
            )
        return RAGAnswer(
            answer=answer,
            sources=tuple(chunks),
        )
