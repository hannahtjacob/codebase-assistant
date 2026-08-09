"""A small, deliberately lexical search engine for code chunks."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass

from chunker import CodeChunk, chunk_file
from scan_repository import scan_repository


TOKEN_PATTERN = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[0-9]+"
)

# These words add little value to code search. This is deliberately a compact,
# understandable list rather than a language model or a linguistic dependency.
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class SearchResult:
    """A code chunk together with its explainable lexical score."""

    chunk: CodeChunk
    score: int
    matched_terms: tuple[str, ...]


def tokenize(text: str) -> list[str]:
    """Tokenize prose and snake_case/camelCase identifiers."""
    tokens = (match.group(0).lower() for match in TOKEN_PATTERN.finditer(text))
    return [token for token in tokens if token not in STOP_WORDS]


def search(
    query: str, chunks: list[CodeChunk], limit: int = 10
) -> list[SearchResult]:
    """Rank chunks by exact occurrences of meaningful query tokens.

    Every occurrence contributes one point. Results with the same score prefer
    chunks matching more distinct terms, then use source position for stable
    ordering. Chunks with no matching tokens are omitted.
    """
    if limit < 0:
        raise ValueError("limit cannot be negative")
    if limit == 0:
        return []

    query_terms = set(tokenize(query))
    if not query_terms:
        return []

    results: list[SearchResult] = []
    for chunk in chunks:
        token_counts = Counter(tokenize(chunk.content))
        matched_terms = tuple(sorted(query_terms & token_counts.keys()))
        score = sum(token_counts[term] for term in matched_terms)
        if score:
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    matched_terms=matched_terms,
                )
            )

    results.sort(
        key=lambda result: (
            -result.score,
            -len(result.matched_terms),
            result.chunk.file_path,
            result.chunk.start_line,
        )
    )
    return results[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="question or keywords to search for")
    parser.add_argument(
        "--repo",
        default="data/repos/requests",
        help="repository to search (default: data/repos/requests)",
    )
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    chunks = [
        chunk
        for source_file in scan_repository(args.repo)
        for chunk in chunk_file(source_file)
    ]
    results = search(args.query, chunks, limit=args.limit)

    print(f"Query tokens: {', '.join(dict.fromkeys(tokenize(args.query))) or '(none)'}")
    print(f"Found {len(results)} matching chunks")
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        print(
            f"{index}. {chunk.file_path}:{chunk.start_line}-{chunk.end_line} "
            f"(score={result.score}, matches={', '.join(result.matched_terms)})"
        )


if __name__ == "__main__":
    main()
