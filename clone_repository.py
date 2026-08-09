"""Clone a remote Git repository into the local repository cache."""

from pathlib import Path
from urllib.parse import urlparse

from git import InvalidGitRepositoryError, Repo


REPOSITORIES_DIR = Path("data/repos")


def _repository_name(url: str) -> str:
    """Return the repository name from an HTTPS or SSH-style Git URL."""
    path = urlparse(url).path if "://" in url else url.rsplit(":", 1)[-1]
    name = Path(path.rstrip("/")).name
    if name.endswith(".git"):
        name = name[:-4]
    if not name:
        raise ValueError(f"Could not determine a repository name from URL: {url}")
    return name


def clone_repository(url: str) -> Path:
    """Clone *url* under ``data/repos`` and return its local path.

    If the repository is already present, its existing working tree is reused.
    No fetch or pull is performed, so this function never changes an existing
    checkout.
    """
    destination = REPOSITORIES_DIR / _repository_name(url)

    if destination.exists():
        try:
            Repo(destination)
        except InvalidGitRepositoryError as error:
            raise FileExistsError(
                f"Clone destination exists but is not a Git repository: {destination}"
            ) from error
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    Repo.clone_from(url, destination)
    return destination


if __name__ == "__main__":
    repository_path = clone_repository("https://github.com/psf/requests")
    print("Repository cloned successfully.")
    print(f"Path: {repository_path.as_posix()}")
