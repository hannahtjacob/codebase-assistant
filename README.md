# Codebase Assistant

Clone the Requests repository with GitPython:

```bash
python3 -m pip install -r requirements.txt
python3 clone_repository.py
```

Repositories are stored under `data/repos/`. Re-running the script reuses an
existing clone rather than downloading it again or overwriting local changes.
