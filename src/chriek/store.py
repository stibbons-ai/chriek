"""SQLite job store. Status changes are conditional updates; a stale writer can't override them."""

import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    ForeignKey,
    String,
    Text,
    create_engine,
    event,
    select,
    update,
)
from sqlalchemy.engine import URL, Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import QueuePool

from chriek.storage import private_database

QUEUED, RUNNING, SUCCEEDED, FAILED = "queued", "running", "succeeded", "failed"
INTERRUPTED_ERROR = "interrupted: the server stopped while this job was running"
_MEMORY = ":memory:"  # SQLite's sentinel for an ephemeral database


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(f"status IN ('{QUEUED}', '{RUNNING}', '{SUCCEEDED}', '{FAILED}')"),
    )

    # `seq` is the persisted FIFO order; `id` is the public UUIDv7.
    seq: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(36), unique=True)
    owner: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), index=True)
    prompt: Mapped[str] = mapped_column(Text)
    image_count: Mapped[int]
    width: Mapped[int]
    height: Mapped[int]
    seed: Mapped[int]
    created_at: Mapped[float]
    started_at: Mapped[float | None]
    finished_at: Mapped[float | None]
    error: Mapped[str | None] = mapped_column(Text)
    run_record: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    images: Mapped[list[Image]] = relationship(order_by="Image.index", lazy="selectin")


class Image(Base):
    """A published image. Rows exist only for succeeded jobs."""

    __tablename__ = "images"

    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), primary_key=True)
    index: Mapped[int] = mapped_column(primary_key=True)
    seed: Mapped[int]


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self._engine = _create_engine(db_path)
        event.listen(self._engine, "connect", _configure_sqlite)
        Base.metadata.create_all(self._engine)
        self._session = sessionmaker(self._engine, expire_on_commit=False)

    def close(self) -> None:
        self._engine.dispose()

    def create(
        self,
        *,
        owner: str,
        prompt: str,
        image_count: int,
        width: int,
        height: int,
        seed: int,
        now: float,
    ) -> Job:
        job = Job(
            id=str(uuid.uuid7()),
            owner=owner,
            status=QUEUED,
            prompt=prompt,
            image_count=image_count,
            width=width,
            height=height,
            seed=seed,
            created_at=now,
            images=[],
        )
        with self._session.begin() as session:
            session.add(job)
        return job

    def get(self, owner: str, job_id: str) -> Job | None:
        with self._session() as session:
            return session.scalars(
                select(Job).where(Job.id == job_id, Job.owner == owner)
            ).one_or_none()

    def claim_next(self, now: float) -> Job | None:
        """Atomically move the oldest queued job to running."""
        oldest = select(Job.seq).where(Job.status == QUEUED).order_by(Job.seq).limit(1)
        with self._session.begin() as session:
            return session.scalars(
                update(Job)
                .where(Job.seq == oldest.scalar_subquery(), Job.status == QUEUED)
                .values(status=RUNNING, started_at=now)
                .returning(Job)
            ).one_or_none()

    def succeed(self, job_id: str, run_record: dict[str, Any], now: float) -> bool:
        with self._session.begin() as session:
            job = self._finish(session, job_id, SUCCEEDED, None, run_record, now)
            if job is None:
                return False
            session.add_all(
                Image(job_id=job_id, index=i, seed=job.seed + i) for i in range(job.image_count)
            )
        return True

    def fail(self, job_id: str, error: str, run_record: dict[str, Any], now: float) -> bool:
        with self._session.begin() as session:
            return self._finish(session, job_id, FAILED, error, run_record, now) is not None

    def fail_interrupted(self, now: float) -> None:
        """Startup recovery: jobs left running by a previous process fail without retry."""
        with self._session.begin() as session:
            session.execute(
                update(Job)
                .where(Job.status == RUNNING)
                .values(status=FAILED, error=INTERRUPTED_ERROR, finished_at=now)
            )

    def is_published(self, job_id: str, index: int) -> bool:
        with self._session() as session:
            return (
                session.scalar(
                    select(Image.index)
                    .join(Job, Job.id == Image.job_id)
                    .where(Image.job_id == job_id, Image.index == index, Job.status == SUCCEEDED)
                )
                is not None
            )

    @staticmethod
    def _finish(session, job_id, status, error, run_record, now) -> Job | None:
        return session.scalars(
            update(Job)
            .where(Job.id == job_id, Job.status == RUNNING)
            .values(status=status, error=error, run_record=run_record, finished_at=now)
            .returning(Job)
        ).one_or_none()


def _create_engine(db_path: Path) -> Engine:
    if str(db_path) == _MEMORY:  # Path("./:memory:") normalizes to ":memory:" too
        # An in-memory database exists only inside its one connection. A single-slot pool
        # keeps that connection for the store's lifetime and lends it to one session at a
        # time, so HTTP and worker threads share the data without interleaving transactions.
        return create_engine(
            URL.create("sqlite", database=_MEMORY),
            poolclass=QueuePool,
            pool_size=1,
            max_overflow=0,
            connect_args={"check_same_thread": False},
        )
    db_path = private_database(db_path)
    # URL.create keeps the filename literal; an f-string URL would parse '?' and '%xx'.
    return create_engine(URL.create("sqlite", database=str(db_path)))


def _configure_sqlite(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
