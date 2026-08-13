"""Ask grounded questions about an indexed repository using local Ollama."""

from __future__ import annotations

import argparse

from llm_provider import DEFAULT_OLLAMA_MODEL, OllamaProvider
from metadata_store import DEFAULT_DATABASE_PATH, MetadataStore
from rag import RAGService
from retriever import Retriever
from semantic_search import DEFAULT_MODEL, load_model
from vector_store import DEFAULT_CHROMA_PATH, ChromaCodeSearch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="retrieved chunks supplied to the LLM (default: 1)",
    )
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--database", default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    repository_id = input("Repository: ").strip()
    print("\nQuestion:")
    question = input().strip()

    vector_search = ChromaCodeSearch(
        persist_path=args.chroma_path,
        model=load_model(args.embedding_model),
        metadata_store=MetadataStore(args.database),
    )
    service = RAGService(
        Retriever(vector_search),
        OllamaProvider(model=args.ollama_model, base_url=args.ollama_url),
    )
    result = service.answer(repository_id, question, args.top_k)
    print(f"\nAnswer:\n\n{result.answer}")


if __name__ == "__main__":
    main()
