"""In-memory semantic search for code chunks using Sentence Transformers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from chunker import CodeChunk, chunk_file
from scan_repository import scan_repository


DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingModel(Protocol):
    """The small part of the SentenceTransformer API used by this module."""

    def encode(self, sentences: str | Sequence[str]) -> object: ...


@dataclass(frozen=True)
class SemanticSearchResult:
    """A code chunk and its cosine similarity to the query."""

    chunk: CodeChunk
    score: float


def load_model(model_name: str = DEFAULT_MODEL) -> EmbeddingModel:
    """Load a cached Sentence Transformer, downloading it on first use."""
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except OSError:
        return SentenceTransformer(model_name)


def embed_texts(
    texts: Sequence[str], model: EmbeddingModel
) -> NDArray[np.float32]:
    """Turn text into a two-dimensional matrix of embedding vectors."""
    embeddings = np.asarray(model.encode(list(texts)), dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("the embedding model must return one vector per text")
    return embeddings


def cosine_similarity(
    query_vector: NDArray[np.floating],
    chunk_vectors: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Calculate cosine similarity between one query and every chunk.

    Cosine similarity is the dot product divided by both vector lengths. A
    value near 1 means the vectors point in a similar direction.
    """
    query = np.asarray(query_vector, dtype=np.float32)
    chunks = np.asarray(chunk_vectors, dtype=np.float32)

    if query.ndim != 1:
        raise ValueError("query_vector must be one-dimensional")
    if chunks.ndim != 2:
        raise ValueError("chunk_vectors must be two-dimensional")
    if chunks.shape[1] != query.shape[0]:
        raise ValueError("query and chunk vectors must have the same dimensions")

    query_length = np.linalg.norm(query)
    chunk_lengths = np.linalg.norm(chunks, axis=1)
    denominators = query_length * chunk_lengths
    dot_products = chunks @ query

    # A zero vector has no direction, so define its similarity as zero.
    return np.divide(
        dot_products,
        denominators,
        out=np.zeros_like(dot_products, dtype=np.float32),
        where=denominators != 0,
    )


def search(
    query: str,
    chunks: Sequence[CodeChunk],
    model: EmbeddingModel,
    limit: int = 5,
) -> list[SemanticSearchResult]:
    """Embed and rank all chunks in memory by cosine similarity."""
    if limit < 0:
        raise ValueError("limit cannot be negative")
    if limit == 0 or not chunks:
        return []

    chunk_vectors = embed_texts([chunk.content for chunk in chunks], model)
    query_vectors = embed_texts([query], model)
    if chunk_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("query and chunk embeddings must have the same dimensions")

    scores = cosine_similarity(query_vectors[0], chunk_vectors)
    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: (
            -float(scores[index]),
            chunks[index].file_path,
            chunks[index].start_line,
        ),
    )
    return [
        SemanticSearchResult(chunks[index], float(scores[index]))
        for index in ranked_indices[:limit]
    ]


def run_tiny_experiment(model: EmbeddingModel) -> None:
    """Show that related wording can be close without exact token matches."""
    sentences = [
        "authenticate a user",
        "calculate payment amount",
        "validate login credentials",
    ]
    query = "check whether the user's password is correct"
    sentence_vectors = embed_texts(sentences, model)
    query_vector = embed_texts([query], model)[0]
    scores = cosine_similarity(query_vector, sentence_vectors)

    print(f"Query: {query}")
    for index in np.argsort(-scores):
        print(f"{scores[index]:.4f}  {sentences[index]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="question to search for")
    parser.add_argument(
        "--repo",
        default="data/repos/requests",
        help="repository to search (default: data/repos/requests)",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--experiment", action="store_true")
    args = parser.parse_args()

    model = load_model(args.model)
    if args.experiment:
        run_tiny_experiment(model)
        return
    if not args.query:
        parser.error("query is required unless --experiment is used")

    chunks = [
        chunk
        for source_file in scan_repository(args.repo)
        for chunk in chunk_file(source_file)
    ]
    results = search(args.query, chunks, model, limit=args.limit)
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        print(
            f"{index}. {chunk.file_path}:{chunk.start_line}-{chunk.end_line} "
            f"(cosine_similarity={result.score:.4f})"
        )


if __name__ == "__main__":
    main()
