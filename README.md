# Codebase Assistant

Clone the Requests repository with GitPython:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python clone_repository.py
```

Repositories are stored under `data/repos/`. Re-running the script reuses an
existing clone rather than downloading it again or overwriting local changes.

Scan a cloned repository for supported source files:

```bash
python scan_repository.py data/repos/requests
```

The scanner supports `.py`, `.js`, `.ts`, `.java`, `.cpp`, and `.h` files. It
ignores dependency, cache, distribution, build, binary, lock, and files larger
than 1 MB.

Split a scanned source file into overlapping 50-line windows:

```python
from chunker import chunk_file
from scan_repository import scan_repository

source_files = scan_repository("data/repos/requests")
chunks = chunk_file(source_files[0], chunk_size=50, overlap=10)
```

Chunks use inclusive, 1-based line numbers. With the default settings, their
ranges are 1–50, 41–90, 81–130, and so on. Chunk IDs are deterministic SHA-256
hashes of their source metadata and content. This phase does not create or use
embeddings.

Search those chunks with exact keyword overlap:

```bash
python keyword_search.py "Where is authentication handled?"
```

The baseline search splits prose and code identifiers into lowercase tokens,
removes common question words, and scores one point per exact token occurrence.
It deliberately has no stemming, synonyms, fuzzy matching, or embeddings, so
its lexical limitations remain visible and measurable.

Search chunks semantically with Sentence Transformers and NumPy:

```bash
python semantic_search.py --experiment
python semantic_search.py "Where are user credentials checked?"
```

The experiment embeds three short sentences and compares them with a password
question. Semantic search uses the same process for `CodeChunk.content`: embed
the question and every chunk, calculate cosine similarity directly with NumPy,
sort the scores, and return the top five. Everything stays in memory; no vector
database or ChromaDB is used.

Persist those embeddings in ChromaDB so they do not need to be recomputed on
every search:

```bash
python vector_store.py index requests --repo data/repos/requests
python vector_store.py search requests "Where are user credentials checked?"
```

The `index` command stores each chunk's embedding and content along with its
chunk ID, file path, line range, language, and repository ID. The `search`
command embeds only the question, filters the collection by repository ID, and
asks Chroma for the five nearest vectors. Data persists under `data/chroma/`.

SQLite remains the right tool for exact structured lookups such as
`SELECT * FROM repositories WHERE id = ?`. ChromaDB serves a different need:
similarity search across high-dimensional embedding vectors.

Run the automated tests with:

```bash
python -m unittest discover -s tests
```
