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

Run the automated tests with:

```bash
python -m unittest discover -s tests
```
