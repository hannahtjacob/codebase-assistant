"""Interactive semantic code search with no LLM involved."""

from __future__ import annotations

import argparse

from metadata_store import DEFAULT_DATABASE_PATH, MetadataStore
from retriever import RetrievedChunk, Retriever
from semantic_search import DEFAULT_MODEL, load_model
from vector_store import DEFAULT_CHROMA_PATH, ChromaCodeSearch


def format_results(results: list[RetrievedChunk]) -> str:
    """Format ranked chunks for terminal output."""
    if not results:
        return "No matching chunks found."

    lines: list[str] = []
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        symbol = f"  {chunk.symbol_type}: {chunk.symbol_name}" if chunk.symbol_name else ""
        lines.extend(
            [
                f"{rank}. {chunk.file_path}:{chunk.start_line}-{chunk.end_line}",
                f"   score={result.score:.4f}{symbol}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--database", default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    repository_id = input("Repository: ").strip()
    print("\nQuestion:")
    question = input().strip()
    if not repository_id:
        parser.error("repository cannot be empty")
    if not question:
        parser.error("question cannot be empty")

    search_engine = ChromaCodeSearch(
        persist_path=args.chroma_path,
        model=load_model(args.model),
        metadata_store=MetadataStore(args.database),
    )
    retriever = Retriever(search_engine)
    results = retriever.retrieve_with_scores(repository_id, question, args.top_k)
    print(f"\nResults:\n\n{format_results(results)}")


if __name__ == "__main__":
    main()
