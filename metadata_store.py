"""Structured repository, chunk, and query metadata stored in SQLite."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from sqlalchemy import ForeignKey, Integer, String, Text, create_engine, delete, event, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from sqlalchemy.pool import NullPool

from chunker import CodeChunk


DEFAULT_DATABASE_PATH = "data/metadata.db"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def initialize_database(database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
    """Create the SQLite schema directly from SQL before the ORM is used."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


class Base(DeclarativeBase):
    pass


class Repository(Base):
    __tablename__ = "repositories"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    url: Mapped[str] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text)
    commit_hash: Mapped[str] = mapped_column(String)
    indexed_at: Mapped[str] = mapped_column(String)
    chunks: Mapped[list[Chunk]] = relationship(cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    repository_id: Mapped[str] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"), index=True
    )
    file_path: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String)
    symbol_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    symbol_type: Mapped[str | None] = mapped_column(String, nullable=True)
    start_line: Mapped[int] = mapped_column(Integer)
    end_line: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)


class QueryHistory(Base):
    __tablename__ = "query_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repository_id: Mapped[str] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"), index=True
    )
    question: Mapped[str] = mapped_column(Text)
    top_k: Mapped[int] = mapped_column(Integer)
    searched_at: Mapped[str] = mapped_column(String)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MetadataStore:
    """SQLAlchemy interface over the manually defined SQLite schema."""

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        initialize_database(database_path)
        self.engine = create_engine(
            f"sqlite:///{Path(database_path)}",
            poolclass=NullPool,
        )
        event.listen(
            self.engine,
            "connect",
            lambda dbapi_connection, _: dbapi_connection.execute(
                "PRAGMA foreign_keys = ON"
            ),
        )

    def save_index(
        self,
        repository_id: str,
        url: str,
        name: str,
        commit_hash: str,
        chunks: Sequence[CodeChunk],
        record_ids: Sequence[str],
    ) -> None:
        """Replace one repository's structured index in a transaction."""
        if len(chunks) != len(record_ids):
            raise ValueError("chunks and record_ids must have the same length")
        with Session(self.engine) as session, session.begin():
            repository = session.get(Repository, repository_id)
            if repository is None:
                repository = Repository(id=repository_id)
                session.add(repository)
            repository.url = url
            repository.name = name
            repository.commit_hash = commit_hash
            repository.indexed_at = _now()

            session.execute(delete(Chunk).where(Chunk.repository_id == repository_id))
            session.add_all(
                Chunk(
                    id=record_id,
                    repository_id=repository_id,
                    file_path=chunk.file_path,
                    language=chunk.language,
                    symbol_name=chunk.symbol_name,
                    symbol_type=chunk.symbol_type,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    content_hash=hashlib.sha256(
                        chunk.content.encode("utf-8")
                    ).hexdigest(),
                    content=chunk.content,
                )
                for chunk, record_id in zip(chunks, record_ids)
            )

    def get_chunks(self, record_ids: Sequence[str]) -> dict[str, CodeChunk]:
        """Fetch structured chunks by their Chroma record IDs."""
        if not record_ids:
            return {}
        with Session(self.engine) as session:
            rows = session.scalars(select(Chunk).where(Chunk.id.in_(record_ids)))
            return {
                row.id: CodeChunk(
                    id=row.id,
                    file_path=row.file_path,
                    language=row.language,
                    start_line=row.start_line,
                    end_line=row.end_line,
                    content=row.content,
                    symbol_name=row.symbol_name,
                    symbol_type=row.symbol_type,
                )
                for row in rows
            }

    def record_query(self, repository_id: str, question: str, top_k: int) -> None:
        with Session(self.engine) as session, session.begin():
            session.add(
                QueryHistory(
                    repository_id=repository_id,
                    question=question,
                    top_k=top_k,
                    searched_at=_now(),
                )
            )
