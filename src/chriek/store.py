"""Job store on SQLite or Postgres. Status changes are conditional updates; a stale writer can't
override them."""

import hashlib
import uuid
from pathlib import Path
from typing import Annotated, Any

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Index,
    String,
    Text,
    create_engine,
    event,
    func,
    insert,
    inspect,
    literal,
    select,
    update,
)
from sqlalchemy.engine import URL, Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.schema import CreateIndex

from chriek.storage import StorageError, private_database

QUEUED, RUNNING, SUCCEEDED, FAILED = "queued", "running", "succeeded", "failed"
INTERRUPTED_ERROR = "interrupted: the server stopped while this job was running"
TIMED_OUT_ERROR = "timed out: the job ran longer than CHRIEK_JOB_TIMEOUT_SECONDS"
MEMORY_DATABASE = ":memory:"  # SQLite's sentinel for an ephemeral database
# Seeds reach 2**32 - 1, which overflows a Postgres INTEGER.
Seed = Annotated[int, mapped_column(BigInteger)]


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(f"status IN ('{QUEUED}', '{RUNNING}', '{SUCCEEDED}', '{FAILED}')"),
        # Covers the per-owner queued count of every submission.
        Index("ix_jobs_owner_status", "owner", "status"),
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
    seed: Mapped[Seed]
    created_at: Mapped[float]
    started_at: Mapped[float | None]
    finished_at: Mapped[float | None]
    error: Mapped[str | None] = mapped_column(Text)
    run_record: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    images: Mapped[list[Image]] = relationship(order_by="Image.index", lazy="selectin")
    references: Mapped[list[Reference]] = relationship(order_by="Reference.index", lazy="selectin")

    @property
    def reference_count(self) -> int:
        return len(self.references)


class Image(Base):
    """A published image. Rows exist only for succeeded jobs."""

    __tablename__ = "images"

    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), primary_key=True)
    index: Mapped[int] = mapped_column(primary_key=True)
    seed: Mapped[Seed]


class Reference(Base):
    """An uploaded reference image associated with an edit job in submission order."""

    __tablename__ = "job_references"

    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), primary_key=True)
    index: Mapped[int] = mapped_column(primary_key=True)


class CommitOutcomeUnknown(Exception):
    """A failed COMMIT call: the database may have committed the job anyway."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"commit outcome unknown for job {job_id}")


class JobStore:
    def __init__(self, database: Path | URL) -> None:
        self._engine = _create_engine(database)
        _rename_draft_reference_table(self._engine)
        Base.metadata.create_all(self._engine)
        # create_all skips existing tables and so their indexes: add indexes introduced
        # after a database file was created. Idempotent; existing rows stay untouched.
        with self._engine.begin() as connection:
            for table in Base.metadata.sorted_tables:
                for index in table.indexes:
                    connection.execute(CreateIndex(index, if_not_exists=True))
        self._session = sessionmaker(self._engine, expire_on_commit=False)

    def close(self) -> None:
        self._engine.dispose()

    def create(
        self,
        *,
        job_id: str | None = None,
        owner: str,
        prompt: str,
        image_count: int,
        width: int,
        height: int,
        seed: int,
        reference_count: int = 0,
        now: float,
        max_queued: int,
    ) -> Job | None:
        """Queue a job unless `owner` already has `max_queued` queued jobs; None if full.

        A queue that is already full when read is rejected by that read alone, without
        locks, so a flood of submissions against a full queue doesn't hold up other writers
        (the worker saving results). That read only rejects; it never admits. Submissions
        that pass it while the queue is near the cap may still be rejected by the count
        inside the one INSERT ... SELECT statement. SQLite runs that statement under its
        write lock. Postgres doesn't: a statement doesn't see rows of transactions still in
        flight, so admission first takes a transaction-scoped advisory lock for `owner`,
        and the statement's fresh READ COMMITTED snapshot then sees every earlier admission
        for that owner. Either way the count sees the rows it races with, and concurrent
        submissions, even through other connections, can't overshoot the cap. Other owners
        and the worker never take that lock. Keep that count inside the statement; an
        unconditional insert after a separate count would let two submissions both see room.
        Reference rows join the admitting transaction, so a visible queued job always has its
        full ordered association.

        Raises `CommitOutcomeUnknown` when the COMMIT call or the session exit after it fails
        (for example a lost connection, or any BaseException): the job may or may not have
        been committed. Any other exception means nothing was committed.
        """
        jobs = Job.__table__
        queued = (
            select(func.count())
            .select_from(jobs)
            .where(jobs.c.owner == owner, jobs.c.status == QUEUED)
        )
        with self._session() as session:
            if session.scalar(queued) >= max_queued:
                return None
        values = {
            "id": job_id or str(uuid.uuid7()),
            "owner": owner,
            "status": QUEUED,
            "prompt": prompt,
            "image_count": image_count,
            "width": width,
            "height": height,
            "seed": seed,
            "created_at": now,
        }
        references = [Reference(job_id=values["id"], index=i) for i in range(reference_count)]
        row = select(*(literal(value, jobs.c[name].type) for name, value in values.items()))
        has_room = queued.correlate(None).scalar_subquery() < max_queued
        # A flag rather than a try around commit(): closing the session after COMMIT can
        # fail too, and that must not read as a rollback either. BaseException too: callers
        # delete the upload on anything but CommitOutcomeUnknown.
        committing = False
        try:
            with self._session() as session:
                if self._engine.dialect.name == "postgresql":
                    session.execute(select(func.pg_advisory_xact_lock(_owner_lock_key(owner))))
                seq = session.scalar(
                    insert(jobs)
                    .from_select(list(values), row.where(has_room))
                    .returning(jobs.c.seq)
                )
                if seq is None:
                    return None
                session.add_all(references)
                session.flush()  # so that commit() below only sends COMMIT
                # Built before COMMIT: failing here must still roll back.
                job = Job(**values, seq=seq, images=[], references=references)
                committing = True
                session.commit()
        except BaseException as exc:
            if committing:
                raise CommitOutcomeUnknown(values["id"]) from exc
            raise
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

    def fail_timed_out(self, job_id: str, timeout_seconds: int, now: float) -> bool:
        """Fail the job if it is still running more than `timeout_seconds` after it started."""
        with self._session.begin() as session:
            result = session.execute(
                update(Job)
                .where(
                    Job.id == job_id,
                    Job.status == RUNNING,
                    Job.started_at < now - timeout_seconds,
                )
                .values(status=FAILED, error=TIMED_OUT_ERROR, finished_at=now)
            )
            return result.rowcount == 1

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


def _rename_draft_reference_table(engine: Engine) -> None:
    """Earlier SQLite drafts named the association table `references`; keep their rows.

    Those drafts predate Postgres support, so a Postgres `references` table isn't ours to rename.
    """
    if engine.dialect.name != "sqlite":
        return
    with engine.begin() as connection:
        tables = set(inspect(connection).get_table_names())
        if "references" not in tables:
            return
        if "job_references" in tables:
            raise StorageError(
                "database has both a 'references' and a 'job_references' table; "
                "merge or remove one before starting chriek"
            )
        connection.exec_driver_sql('ALTER TABLE "references" RENAME TO job_references')


def _create_engine(database: Path | URL) -> Engine:
    if isinstance(database, URL):
        # The worker holds no connection while rendering; pre-ping replaces one the server dropped
        # meanwhile (restart, failover, idle timeout), but can't save a result while the database
        # stays unreachable. A forced UTF-8 client encoding overrides URL and PGCLIENTENCODING
        # settings, so every prompt and error text reaches a UTF8 database. READ COMMITTED
        # overrides a stricter server default: queue admission needs each statement's fresh
        # snapshot (see JobStore.create).
        return create_engine(
            database.set(drivername="postgresql+psycopg"),
            pool_pre_ping=True,
            connect_args={"client_encoding": "utf8"},
            isolation_level="READ COMMITTED",
        )
    return _sqlite_engine(database)


def _owner_lock_key(owner: str) -> int:
    """Stable signed 64-bit advisory-lock key; a collision only serializes two owners."""
    digest = hashlib.blake2b(owner.encode(errors="surrogatepass"), digest_size=8).digest()
    return int.from_bytes(digest, signed=True)


def _sqlite_engine(path: Path) -> Engine:
    if str(path) == MEMORY_DATABASE:  # Path("./:memory:") normalizes to ":memory:" too
        # An in-memory database exists only inside its one connection. A single-slot pool
        # keeps that connection for the store's lifetime and lends it to one session at a
        # time, so HTTP and worker threads share the data without interleaving transactions.
        engine = create_engine(
            URL.create("sqlite", database=MEMORY_DATABASE),
            poolclass=QueuePool,
            pool_size=1,
            max_overflow=0,
            connect_args={"check_same_thread": False},
        )
    else:
        path = private_database(path)
        # URL.create keeps the filename literal; an f-string URL would parse '?' and '%xx'.
        engine = create_engine(URL.create("sqlite", database=str(path)))
    event.listen(engine, "connect", _configure_sqlite)
    return engine


def _configure_sqlite(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
