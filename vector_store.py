"""Persistent semantic search for code chunks using ChromaDB."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from chunker import CodeChunk, chunk_file
from scan_repository import scan_repository
from semantic_search import (
    DEFAULT_MODEL,
    EmbeddingModel,
    embed_texts,
    load_model,
)


DEFAULT_CHROMA_PATH = "data/chroma"
DEFAULT_COLLECTION = "code_chunks"


@dataclass(frozen=True)
class VectorSearchResult:
    """A persisted code chunk and its cosine similarity to a question."""

    chunk: CodeChunk
    repository_id: str
    score: float


class ChromaCodeSearch:
    """Index and search code embeddings in a persistent Chroma collection."""

    def __init__(
        self,
        persist_path: str | Path = DEFAULT_CHROMA_PATH,
        model: EmbeddingModel | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        client: Any | None = None,
    ) -> None:
        if client is None:
            import chromadb

            client = chromadb.PersistentClient(path=str(persist_path))

        self.model = model or load_model()
        self.collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

    @staticmethod
    def _record_id(repository_id: str, chunk_id: str) -> str:
        """Create a Chroma ID unique across all indexed repositories."""
        identity = f"{repository_id}\0{chunk_id}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def index_chunks(
        self, chunks: Sequence[CodeChunk], repository_id: str
    ) -> int:
        """Embed and persist chunks for one repository, updating existing IDs."""
        if not repository_id.strip():
            raise ValueError("repository_id cannot be empty")
        if not chunks:
            return 0

        embeddings = embed_texts([chunk.content for chunk in chunks], self.model)
        self.collection.upsert(
            ids=[self._record_id(repository_id, chunk.id) for chunk in chunks],
            embeddings=embeddings.tolist(),
            documents=[chunk.content for chunk in chunks],
            metadatas=[
                {
                    "chunk_id": chunk.id,
                    "file_path": chunk.file_path,
                    "language": chunk.language,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "repository_id": repository_id,
                }
                for chunk in chunks
            ],
        )
        return len(chunks)

    def search(
        self, repository_id: str, question: str, top_k: int = 5
    ) -> list[VectorSearchResult]:
        """Return the nearest persisted chunks belonging to one repository."""
        if not repository_id.strip():
            raise ValueError("repository_id cannot be empty")
        if top_k < 0:
            raise ValueError("top_k cannot be negative")
        if top_k == 0:
            return []

        query_embedding = embed_texts([question], self.model)[0]
        response = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            where={"repository_id": repository_id},
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = response.get("documents") or [[]]
        metadatas = response.get("metadatas") or [[]]
        distances = response.get("distances") or [[]]
        results: list[VectorSearchResult] = []
        for document, metadata, distance in zip(
            documents[0], metadatas[0], distances[0]
        ):
            if document is None or metadata is None or distance is None:
                continue
            chunk = CodeChunk(
                id=str(metadata["chunk_id"]),
                file_path=str(metadata["file_path"]),
                language=str(metadata["language"]),
                start_line=int(metadata["start_line"]),
                end_line=int(metadata["end_line"]),
                content=document,
            )
            results.append(
                VectorSearchResult(
                    chunk=chunk,
                    repository_id=str(metadata["repository_id"]),
                    # A cosine distance of 0 means identical direction.
                    score=1.0 - float(distance),
                )
            )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="embed and store a repository")
    index_parser.add_argument("repository_id")
    index_parser.add_argument("--repo", default="data/repos/requests")

    search_parser = subparsers.add_parser("search", help="search stored embeddings")
    search_parser.add_argument("repository_id")
    search_parser.add_argument("question")
    search_parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    store = ChromaCodeSearch(
        persist_path=args.chroma_path,
        model=load_model(args.model),
    )
    if args.command == "index":
        chunks = [
            chunk
            for source_file in scan_repository(args.repo)
            for chunk in chunk_file(source_file)
        ]
        count = store.index_chunks(chunks, args.repository_id)
        print(f"Indexed {count} chunks for repository {args.repository_id!r}")
        return

    results = store.search(args.repository_id, args.question, args.top_k)
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        print(
            f"{index}. {chunk.file_path}:{chunk.start_line}-{chunk.end_line} "
            f"(cosine_similarity={result.score:.4f})"
        )


if __name__ == "__main__":
    main()
