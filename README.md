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

Run the automated tests with:

```bash
python -m unittest discover -s tests
```
