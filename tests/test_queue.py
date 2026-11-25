"""Property 4: one job at a time in persisted FIFO order; restarts fail running jobs only."""

import io
import logging
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from conftest import (
    ALICE,
    BOB,
    POSTGRES_URL,
    SIGNING_SECRET,
    CountingBackend,
    FailingBackend,
    FakeClock,
    GatedBackend,
    base_env,
    connect_db,
    db_dump,
    make_image_bytes,
    running,
    submit,
    tree_state,
    wait_for,
)
from PIL import Image
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from chriek.api import WORKER_STOPPED
from chriek.images import LocalImages, S3Images, reference_path
from chriek.signing import UrlSigner
from chriek.store import JobStore
from chriek.worker import Worker


def status(client, job_id, headers=ALICE):
    return client.get(f"/jobs/{job_id}", headers=headers).json()["status"]


def test_jobs_run_one_at_a_time_in_submission_order(tmp_path):
    backend = GatedBackend()
    with running(tmp_path, backend=backend) as client:
        first = submit(client, prompt="first", width=16, height=16)
        assert backend.rendering.wait(5)
        second = submit(client, headers=BOB, prompt="second", width=16, height=16)
        third = submit(client, prompt="third", width=16, height=16)

        assert status(client, first["id"]) == "running"
        assert status(client, second["id"], headers=BOB) == "queued"
        assert status(client, third["id"]) == "queued"

        backend.release()
        bodies = [
            wait_for(client, first["id"]),
            wait_for(client, second["id"], headers=BOB),
            wait_for(client, third["id"]),
        ]

    assert [body["status"] for body in bodies] == ["succeeded"] * 3
    assert backend.prompts == ["first", "second", "third"]
    assert backend.max_active == 1
    for earlier, later in zip(bodies, bodies[1:], strict=False):
        assert earlier["finished_at"] <= later["started_at"]


@pytest.mark.local_storage
def test_restart_fails_running_job_and_runs_queued_job(tmp_path):
    crashed_backend = GatedBackend()
    stale_clock = FakeClock()
    with running(
        tmp_path, backend=crashed_backend, clock=stale_clock, CHRIEK_JOB_TIMEOUT_SECONDS="60"
    ) as before:
        running_job = submit(before, prompt="interrupted", width=16, height=16)
        assert crashed_backend.rendering.wait(5)
        queued_job = submit(before, prompt="survivor", image_count=2, seed=8, width=16, height=16)
        assert status(before, running_job["id"]) == "running"

        # A second process starts on the same database and storage, as after a crash.
        with running(tmp_path) as after:
            interrupted = after.get(f"/jobs/{running_job['id']}", headers=ALICE).json()
            assert interrupted["status"] == "failed"
            assert interrupted["error"] == (
                "interrupted: the server stopped while this job was running"
            )
            assert interrupted["images"] == []

            survivor = wait_for(after, queued_job["id"])
            assert survivor["status"] == "succeeded"
            assert [image["seed"] for image in survivor["images"]] == [8, 9]
            assert after.get(survivor["images"][1]["url"]).status_code == 200
            assert after.get(f"/jobs/{queued_job['id']}", headers=BOB).status_code == 404

            # Let the stale process finish its render past its own deadline; stopping it waits
            # for that. Its timeout check must not overwrite the interrupted failure.
            stale_clock.advance(61)
            crashed_backend.release()

    assert not [t for t in threading.enumerate() if t.name == "chriek-worker"], "worker not drained"
    # The stale process's late result must not resurrect or relabel the failed job.
    with running(tmp_path) as later:
        stale = later.get(f"/jobs/{running_job['id']}", headers=ALICE).json()
        assert stale["status"] == "failed"
        assert stale["error"] == "interrupted: the server stopped while this job was running"
        assert stale["images"] == []
    assert crashed_backend.prompts == ["interrupted"]
    assert not (tmp_path / "images" / running_job["id"]).exists()


@pytest.mark.local_storage
def test_jobs_and_downloads_survive_a_clean_restart(tmp_path):
    with running(tmp_path) as before:
        body = wait_for(before, submit(before, width=16, height=16)["id"])
        assert body["status"] == "succeeded"

    with running(tmp_path) as after:
        again = after.get(f"/jobs/{body['id']}", headers=ALICE).json()
        assert again["status"] == "succeeded"
        assert again["params"] == body["params"]
        assert after.get(again["images"][0]["url"]).status_code == 200


def test_boundary_seeds_and_precise_timestamps_survive_a_restart(tmp_path):
    # The largest seeds overflow a 32-bit INTEGER; microseconds need double precision.
    clock = FakeClock(1_800_000_000.123457)
    with running(tmp_path, clock=clock) as before:
        submitted = submit(before, image_count=2, seed=2**32 - 2, width=16, height=16)
        body = wait_for(before, submitted["id"])
        assert body["status"] == "succeeded", body

    with running(tmp_path, clock=clock) as after:
        again = after.get(f"/jobs/{body['id']}", headers=ALICE).json()

    assert again["params"]["seed"] == 2**32 - 2
    assert [image["seed"] for image in again["images"]] == [2**32 - 2, 2**32 - 1]
    expected = datetime.fromtimestamp(clock.now, UTC).isoformat()
    assert expected.endswith(".123457+00:00")
    assert [again[key] for key in ("created_at", "started_at", "finished_at")] == [expected] * 3


@pytest.mark.skipif(POSTGRES_URL is None, reason="terminates a Postgres server connection")
def test_connection_dropped_during_a_render_does_not_stop_the_worker(
    tmp_path, database, postgres_admin
):
    backend = GatedBackend()
    with running(tmp_path, backend=backend) as client:
        first = submit(client, prompt="first", width=16, height=16)
        assert backend.rendering.wait(5)

        # Act: the server drops every idle pooled connection of this test's own database.
        with postgres_admin.connect() as admin:
            dropped = admin.execute(
                text(
                    "SELECT count(*) FILTER (WHERE pg_terminate_backend(pid)) "
                    "FROM pg_stat_activity WHERE datname = :name AND state = 'idle'"
                ),
                {"name": database},
            ).scalar_one()
        assert dropped >= 1
        backend.release()

        assert wait_for(client, first["id"])["status"] == "succeeded"
        second = submit(client, prompt="second", width=16, height=16)
        assert wait_for(client, second["id"])["status"] == "succeeded"
        assert client.get("/health").json() == {"status": "ok"}


# The schema created by chriek before Postgres support (base 06f0fbe), reflowed.
PRE_POSTGRES_SQLITE_SCHEMA = """
CREATE TABLE jobs (
    seq INTEGER NOT NULL, id VARCHAR(36) NOT NULL, owner VARCHAR(64) NOT NULL,
    status VARCHAR(16) NOT NULL, prompt TEXT NOT NULL, image_count INTEGER NOT NULL,
    width INTEGER NOT NULL, height INTEGER NOT NULL, seed INTEGER NOT NULL,
    created_at DOUBLE NOT NULL, started_at DOUBLE, finished_at DOUBLE, error TEXT,
    run_record JSON, PRIMARY KEY (seq),
    CHECK (status IN ('queued', 'running', 'succeeded', 'failed')), UNIQUE (id)
);
CREATE INDEX ix_jobs_status ON jobs (status);
CREATE TABLE images (
    job_id VARCHAR(36) NOT NULL, "index" INTEGER NOT NULL, seed INTEGER NOT NULL,
    PRIMARY KEY (job_id, "index"), FOREIGN KEY(job_id) REFERENCES jobs (id)
);
"""


@pytest.mark.sqlite
@pytest.mark.local_storage
def test_existing_sqlite_database_keeps_working(tmp_path):
    db_path = tmp_path / "db" / "chriek.db"
    db_path.parent.mkdir(mode=0o700)
    db_path.touch(mode=0o600)  # chriek refuses existing data that other users can access
    job_id = str(uuid.uuid7())
    with sqlite3.connect(db_path) as connection:
        connection.executescript(PRE_POSTGRES_SQLITE_SCHEMA)
        connection.execute(
            "INSERT INTO jobs (id, owner, status, prompt, image_count, width, height, seed,"
            " created_at) VALUES (?, 'alice', 'queued', 'old', 2, 16, 16, ?, 1800000000.5)",
            (job_id, 2**32 - 2),
        )
    connection.close()

    with running(tmp_path) as client:
        old = wait_for(client, job_id)
        assert old["status"] == "succeeded", old
        assert [image["seed"] for image in old["images"]] == [2**32 - 2, 2**32 - 1]
        assert client.get(old["images"][1]["url"]).status_code == 200
        assert client.get(f"/jobs/{job_id}", headers=BOB).status_code == 404
        new = wait_for(client, submit(client, width=16, height=16)["id"])
        assert new["status"] == "succeeded", new


@pytest.mark.sqlite
@pytest.mark.parametrize("filename", ["chriek?home.db", "chriek%20home.db"])
def test_database_path_is_used_literally_across_restart(tmp_path, filename):
    db_dir = tmp_path / "db"
    env = {"CHRIEK_DB_PATH": str(db_dir / filename)}
    with running(tmp_path, **env) as before:
        body = wait_for(before, submit(before, width=16, height=16)["id"])
        assert body["status"] == "succeeded"

    with running(tmp_path, **env) as after:
        again = after.get(f"/jobs/{body['id']}", headers=ALICE).json()
        assert again["status"] == "succeeded"
        assert again["params"] == body["params"]

    assert sorted(path.name for path in db_dir.iterdir() if path.is_file()) == [filename]


@pytest.mark.sqlite
@pytest.mark.parametrize("sentinel", [":memory:", "./:memory:"])
@pytest.mark.local_storage
def test_memory_database_serves_one_lifetime_and_resets_on_restart(tmp_path, monkeypatch, sentinel):
    # A shared (0755) working directory: the database checks must skip it, not refuse it.
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    cwd.chmod(0o755)
    before_cwd = tree_state(cwd)
    monkeypatch.chdir(cwd)
    with running(tmp_path, CHRIEK_DB_PATH=sentinel) as before:
        body = wait_for(before, submit(before, image_count=2, seed=4, width=16, height=16)["id"])
        assert body["status"] == "succeeded"
        assert [image["seed"] for image in body["images"]] == [4, 5]
        download = before.get(body["images"][1]["url"])
        assert download.status_code == 200
        assert Image.open(io.BytesIO(download.content)).text["Seed"] == "5"
        assert before.get(f"/jobs/{body['id']}", headers=BOB).status_code == 404

    with running(tmp_path, CHRIEK_DB_PATH=sentinel) as after:
        assert after.get(f"/jobs/{body['id']}", headers=ALICE).status_code == 404
        assert after.get(body["images"][1]["url"]).status_code == 404
        assert wait_for(after, submit(after, width=16, height=16)["id"])["status"] == "succeeded"

    assert sorted(path.name for path in tmp_path.iterdir()) == ["cwd", "images"]
    assert tree_state(cwd) == before_cwd


@pytest.mark.sqlite
def test_memory_database_keeps_every_write_under_concurrent_polling(tmp_path):
    backend = GatedBackend()
    with running(tmp_path, backend=backend, CHRIEK_DB_PATH=":memory:") as client:
        first = submit(client, prompt="first", width=16, height=16)
        assert backend.rendering.wait(5)

        def submit_and_poll(owner: str, headers: dict[str, str]) -> list[str]:
            ids = []
            for n in range(5):
                job = submit(client, headers=headers, prompt=f"{owner}-{n}", width=16, height=16)
                ids.append(job["id"])
                for job_id in ids:
                    assert client.get(f"/jobs/{job_id}", headers=headers).status_code == 200
            return ids

        # Submissions and polls from two threads overlap the worker's claim/finish writes.
        with ThreadPoolExecutor(2) as pool:
            alice = pool.submit(submit_and_poll, "alice", ALICE)
            bob = pool.submit(submit_and_poll, "bob", BOB)
            backend.release()
            ids = {"alice": alice.result(timeout=30), "bob": bob.result(timeout=30)}

        bodies = [wait_for(client, first["id"])]
        bodies += [wait_for(client, job_id) for job_id in ids["alice"]]
        bodies += [wait_for(client, job_id, headers=BOB) for job_id in ids["bob"]]

    assert [body["status"] for body in bodies] == ["succeeded"] * 11
    assert all(len(body["images"]) == 1 for body in bodies)
    assert sorted(backend.prompts) == sorted(body["params"]["prompt"] for body in bodies)
    assert backend.max_active == 1


@pytest.mark.parametrize(
    ("store_method", "first_render_fails"),
    [("succeed", False), ("fail", True)],
    ids=["success-write-fails", "failure-write-fails"],
)
def test_unsaved_job_result_stops_the_worker_and_rejects_jobs_until_restart(
    tmp_path, monkeypatch, caplog, store_method, first_render_fails
):
    # Arrange: the first final-state write raises, as if the database were locked or the disk full.
    # It waits until the second job is queued, so that job is admitted before the failure.
    original = getattr(JobStore, store_method)
    calls = []
    second_queued = threading.Event()

    def broken_once(self, *args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            assert second_queued.wait(10), "second job never queued"
            raise OperationalError("COMMIT", {}, RuntimeError("database is locked"))
        return original(self, *args, **kwargs)

    notifies = []
    original_notify = Worker.notify

    def counting_notify(self):
        notifies.append(self)
        original_notify(self)

    monkeypatch.setattr(JobStore, store_method, broken_once)
    monkeypatch.setattr(Worker, "notify", counting_notify)
    backend = FailingBackend(fail_index=0) if first_render_fails else CountingBackend()
    job = {"prompt": "rejected", "width": 16, "height": 16}

    # Cap 1: once the first job runs, alice's queued second job fills her queue; bob's stays empty.
    with running(tmp_path, backend=backend, CHRIEK_MAX_QUEUED_JOBS="1") as before:
        first = submit(before, prompt="first", width=16, height=16)
        wait_for(before, first["id"], statuses={"running"})
        second = submit(before, prompt="second", width=16, height=16)
        assert before.post("/jobs", headers=ALICE, json=job).status_code == 429
        second_queued.set()

        deadline = time.monotonic() + 10
        while (health := before.get("/health")).status_code != 503:
            assert time.monotonic() < deadline, "worker never reported its failure"
            time.sleep(0.01)
        stopped = {"status": "error", "detail": WORKER_STOPPED}
        assert health.json() == stopped
        stuck = before.get(f"/jobs/{first['id']}", headers=ALICE).json()
        assert (stuck["status"], stuck["images"], stuck["error"]) == ("running", [], None)
        assert status(before, second["id"]) == "queued"
        persisted, notified = db_dump(tmp_path), len(notifies)

        # Act/assert: valid submissions from any owner, full queue or not, are refused with
        # nothing stored; a stopped worker's 503 takes precedence over a full queue's 429.
        for headers in (ALICE, BOB):
            rejected = before.post("/jobs", headers=headers, json=job)
            assert rejected.status_code == 503
            assert rejected.json() == {"detail": WORKER_STOPPED}
            assert "location" not in rejected.headers
        # Authentication, validation and owner isolation keep their precedence.
        for headers in ({}, {"Authorization": "Bearer nope"}):
            assert before.post("/jobs", headers=headers, json=job).status_code == 401
        for invalid in ({"width": 15}, {"image_count": True}):
            assert before.post("/jobs", headers=ALICE, json=job | invalid).status_code == 422
        json_headers = ALICE | {"Content-Type": "application/json"}
        assert (
            before.post("/jobs", headers=json_headers, content=b'{"x": "\xff"}').status_code == 400
        )
        other_owner = before.get(f"/jobs/{second['id']}", headers=BOB)
        missing = before.get(f"/jobs/{uuid.uuid7()}", headers=BOB)
        assert (other_owner.status_code, other_owner.json()) == (404, missing.json())

        assert (db_dump(tmp_path), len(notifies)) == (persisted, notified)
        assert before.get("/health").json() == stopped
        assert backend.renders == 1, "no job may run after an unsaved result"

    assert any(
        isinstance(record.exc_info[1], OperationalError)
        for record in caplog.records
        if record.levelno == logging.ERROR and record.exc_info
    ), "the original database error must be logged"

    restarted = CountingBackend()
    with running(tmp_path, backend=restarted) as after:
        assert after.get("/health").json() == {"status": "ok"}
        interrupted = after.get(f"/jobs/{first['id']}", headers=ALICE).json()
        assert interrupted["status"] == "failed"
        assert interrupted["error"] == "interrupted: the server stopped while this job was running"
        assert (interrupted["images"], interrupted["run"]) == ([], None)
        assert wait_for(after, second["id"])["status"] == "succeeded"
    assert restarted.renders == 1
    with connect_db(tmp_path) as db:
        stored = db.scalars(text("SELECT id FROM jobs ORDER BY seq")).all()
    assert stored == [first["id"], second["id"]], "a rejected job was persisted"


class StepBackend(GatedBackend):
    """Gated backend whose `release()` lets exactly one render through; `started` counts
    renders waiting at the gate."""

    def __init__(self) -> None:
        super().__init__()
        self._steps = threading.Semaphore(0)
        self.started = threading.Semaphore(0)
        super().release()  # the base gate stays open; `_steps` gates instead

    def release(self) -> None:
        self._steps.release()

    def render(self, prompt, width, height, seed, references=()):
        self.started.release()
        assert self._steps.acquire(timeout=10), "gated render never released"
        return super().render(prompt, width, height, seed, references)


TIMED_OUT = "timed out: the job ran longer than CHRIEK_JOB_TIMEOUT_SECONDS"


@pytest.mark.local_storage
def test_stuck_job_times_out_and_its_late_result_is_discarded(tmp_path):
    clock = FakeClock()
    backend = StepBackend()
    with running(tmp_path, backend=backend, clock=clock, CHRIEK_JOB_TIMEOUT_SECONDS="60") as client:
        stuck = submit(client, prompt="stuck", width=16, height=16)
        assert backend.started.acquire(timeout=5)
        successor = submit(client, prompt="successor", width=16, height=16)

        clock.advance(60)  # exactly at the deadline: still running
        assert status(client, stuck["id"]) == "running"
        clock.advance(1)
        assert client.get(f"/jobs/{stuck['id']}", headers=BOB).status_code == 404
        clock.advance(1)  # a write by Bob's poll would carry the earlier finished_at
        failed = client.get(f"/jobs/{stuck['id']}", headers=ALICE).json()
        assert (failed["status"], failed["error"], failed["images"]) == ("failed", TIMED_OUT, [])
        assert failed["finished_at"] == datetime.fromtimestamp(clock.now, UTC).isoformat()
        # Timing out frees nothing: the successor waits for the stuck render, never beside it.
        assert status(client, successor["id"]) == "queued"

        backend.release()
        assert backend.started.acquire(timeout=5)  # successor claimed after 62 s queued
        clock.advance(30)  # 92 s since submission, 30 s since claim
        # Queued time does not count: only the time since the successor's claim does.
        assert status(client, successor["id"]) == "running"
        backend.release()
        done = wait_for(client, successor["id"])
        assert done["status"] == "succeeded", done
        assert client.get(done["images"][0]["url"]).status_code == 200
        late = client.get(f"/jobs/{stuck['id']}", headers=ALICE).json()
        assert late == failed
        assert client.get(signed_url(stuck["id"], clock)).status_code == 404

    assert backend.max_active == 1
    assert not (tmp_path / "images" / stuck["id"]).exists()


@pytest.mark.sqlite  # holds SQLite's write lock; reads still pass under WAL
def test_overdue_poll_under_a_write_lock_returns_running_then_fails_after_release(tmp_path, caplog):
    clock = FakeClock()
    backend = StepBackend()
    with running(tmp_path, backend=backend, clock=clock, CHRIEK_JOB_TIMEOUT_SECONDS="60") as client:
        stuck = submit(client, prompt="stuck", width=16, height=16)
        assert backend.started.acquire(timeout=5)
        clock.advance(61)

        writer = sqlite3.connect(
            base_env(tmp_path)["CHRIEK_DB_PATH"], timeout=0, isolation_level=None
        )
        try:
            writer.execute("BEGIN IMMEDIATE")
            with caplog.at_level(logging.ERROR, logger="chriek.api"):
                held = client.get(f"/jobs/{stuck['id']}", headers=ALICE)
        finally:
            writer.close()
        # Only the timeout write failed: the readable state comes back, details only in the log.
        assert held.status_code == 200
        assert (held.json()["status"], held.json()["error"]) == ("running", None)
        assert "locked" not in held.text
        assert f"could not time out job {stuck['id']}" in caplog.text
        assert "database is locked" in caplog.text

        polled = client.get(f"/jobs/{stuck['id']}", headers=ALICE)
        assert polled.status_code == 200
        assert (polled.json()["status"], polled.json()["error"]) == ("failed", TIMED_OUT)
        backend.release()


class LateBackend(GatedBackend):
    """Gated render that may fail once, after release; then every clock read advances 1 s."""

    def __init__(self, clock: TickingClock, fails: bool) -> None:
        super().__init__()
        self.clock = clock
        self.fails = fails

    def render(self, prompt, width, height, seed, references=()):
        image = super().render(prompt, width, height, seed, references)
        self.clock.tick = 1
        fails, self.fails = self.fails, False
        if fails:
            raise RuntimeError("render failed after release")
        return image


class TickingClock(FakeClock):
    tick = 0

    def __call__(self) -> float:
        now = self.now
        self.now += self.tick
        return now


def wait_for_worker(tmp_path, job_id: str) -> None:
    """Wait for the worker's own final write, read from the database so no poll can time out."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with connect_db(tmp_path) as db:
            query = text("SELECT status FROM jobs WHERE id = :id")
            if db.scalar(query, {"id": job_id}) not in ("queued", "running"):
                return
        time.sleep(0.01)
    raise AssertionError(f"worker never finished job {job_id}")


@pytest.mark.parametrize("fails", [False, True], ids=["late-success", "late-failure"])
def test_result_finished_after_the_deadline_without_a_poll_fails_the_job(tmp_path, fails):
    clock = TickingClock()
    backend = LateBackend(clock, fails)
    with running(tmp_path, backend=backend, clock=clock, CHRIEK_JOB_TIMEOUT_SECONDS="60") as client:
        job = submit(client, width=16, height=16)
        assert backend.rendering.wait(5)
        clock.advance(61)
        backend.release()  # nobody polls before the worker finishes
        wait_for_worker(tmp_path, job["id"])

        body = client.get(f"/jobs/{job['id']}", headers=ALICE).json()
        assert (body["status"], body["error"]) == ("failed", TIMED_OUT)
        assert (body["run"], body["images"]) == (None, [])
        assert client.get(signed_url(job["id"], clock)).status_code == 404
        assert wait_for(client, submit(client, width=16, height=16)["id"])["status"] == "succeeded"

    assert not (tmp_path / "images" / job["id"]).exists()


@pytest.mark.parametrize(
    ("fails", "outcome"),
    [(False, ("succeeded", None)), (True, ("failed", "image 1 of 1 failed"))],
    ids=["success", "failure"],
)
@pytest.mark.local_storage
def test_result_finished_exactly_at_the_deadline_keeps_its_outcome(tmp_path, fails, outcome):
    # Every clock read after the render advances 1 s, so the worker must decide and persist
    # with one completion time: within the limit, never a later one past it.
    clock = TickingClock()
    backend = LateBackend(clock, fails)
    with running(tmp_path, backend=backend, clock=clock, CHRIEK_JOB_TIMEOUT_SECONDS="60") as client:
        job = submit(client, width=16, height=16)
        assert backend.rendering.wait(5)
        clock.advance(60)
        backend.release()
        wait_for_worker(tmp_path, job["id"])

        body = client.get(f"/jobs/{job['id']}", headers=ALICE).json()
        assert (body["status"], body["error"]) == outcome
        elapsed = datetime.fromisoformat(body["finished_at"]) - datetime.fromisoformat(
            body["started_at"]
        )
        assert elapsed.total_seconds() == 60
        assert len(body["images"]) == (0 if fails else 1)
        if not fails:
            assert client.get(body["images"][0]["url"]).status_code == 200


@pytest.mark.parametrize(
    ("elapsed", "error"),
    [(59, "image 1 of 1 failed"), (61, TIMED_OUT)],
    ids=["failure-in-time", "late-failure"],
)
def test_slow_output_cleanup_never_decides_whether_a_render_failure_timed_out(
    tmp_path, monkeypatch, elapsed, error
):
    clock = TickingClock()
    backend = LateBackend(clock, fails=True)
    for images in (LocalImages, S3Images):

        def slow_remove(self, job_id, count, remove=images.remove_outputs):
            clock.advance(30)  # the cleanup alone outlasts the deadline
            remove(self, job_id, count)

        monkeypatch.setattr(images, "remove_outputs", slow_remove)
    with running(tmp_path, backend=backend, clock=clock, CHRIEK_JOB_TIMEOUT_SECONDS="60") as client:
        job = submit(client, width=16, height=16)
        assert backend.rendering.wait(5)
        clock.advance(elapsed)
        backend.release()  # nobody polls before the worker finishes
        wait_for_worker(tmp_path, job["id"])
        body = client.get(f"/jobs/{job['id']}", headers=ALICE).json()
    assert (body["status"], body["error"], body["images"]) == ("failed", error, [])


@pytest.mark.parametrize(
    ("polled", "fails"),
    [(True, False), (False, False), (False, True)],
    ids=["poll-then-late-success", "worker-late-success", "worker-late-failure"],
)
@pytest.mark.local_storage
def test_timed_out_edit_job_keeps_its_references_and_never_publishes_outputs(
    tmp_path, polled, fails
):
    clock = TickingClock()
    backend = LateBackend(clock, fails)
    refs = [("references", (f"{i}.png", make_image_bytes((i, 0, 0)), "image/png")) for i in (1, 2)]
    with running(tmp_path, backend=backend, clock=clock, CHRIEK_JOB_TIMEOUT_SECONDS="60") as client:
        form = {"prompt": "edit", "width": "16", "height": "16"}
        response = client.post("/jobs", headers=ALICE, data=form, files=refs)
        assert response.status_code == 202, response.text
        job = response.json()
        assert backend.rendering.wait(5)
        clock.advance(61)
        if polled:
            assert status(client, job["id"]) == "failed"
        backend.release()
        wait_for_worker(tmp_path, job["id"])
        assert backend.prompts == ["edit"]

        body = client.get(f"/jobs/{job['id']}", headers=ALICE).json()
        assert (body["status"], body["error"]) == ("failed", TIMED_OUT)
        assert (body["run"], body["images"], body["params"]["reference_count"]) == (None, [], 2)
        assert client.get(signed_url(job["id"], clock)).status_code == 404
        assert client.get(f"/jobs/{job['id']}", headers=BOB).status_code == 404
        assert wait_for(client, submit(client, width=16, height=16)["id"])["status"] == "succeeded"

    # The private references outlive the timeout; no output was left beside them.
    stored = tmp_path / "images" / job["id"]
    assert sorted(p.relative_to(stored).as_posix() for p in stored.rglob("*.png")) == [
        "references/0.png",
        "references/1.png",
    ]


@pytest.mark.parametrize(
    ("tick", "outcome"),
    [
        (31, ("failed", TIMED_OUT, False)),
        (30, ("failed", "reference 2 of 2 could not be loaded", True)),
    ],
    ids=["past-deadline", "exact-deadline"],
)
@pytest.mark.local_storage
def test_reference_load_failure_decides_and_persists_one_completion_time(tmp_path, tick, outcome):
    clock = TickingClock()
    backend = GatedBackend()
    refs = [("references", (f"{i}.png", make_image_bytes((i, 0, 0)), "image/png")) for i in (1, 2)]
    with running(tmp_path, backend=backend, clock=clock, CHRIEK_JOB_TIMEOUT_SECONDS="60") as client:
        submit(client, prompt="hold", width=16, height=16)
        assert backend.rendering.wait(5)
        form = {"prompt": "edit", "width": "16", "height": "16"}
        response = client.post("/jobs", headers=ALICE, data=form, files=refs)
        assert response.status_code == 202, response.text
        job = response.json()
        reference_path(tmp_path / "images", job["id"], 1).unlink()
        # From here each worker clock read advances `tick`: claim T, start T+tick, failure T+2*tick.
        clock.tick = tick
        backend.release()
        wait_for_worker(tmp_path, job["id"])
        clock.tick = 0

        body = client.get(f"/jobs/{job['id']}", headers=ALICE).json()
        assert (body["status"], body["error"], body["run"] is not None) == outcome
        assert (body["images"], body["params"]["reference_count"]) == ([], 2)
        elapsed = datetime.fromisoformat(body["finished_at"]) - datetime.fromisoformat(
            body["started_at"]
        )
        assert elapsed.total_seconds() == 2 * tick
        assert backend.prompts == ["hold"]


def signed_url(job_id: str, clock: FakeClock) -> str:
    """A valid, unexpired download URL for image 0, as if one had been issued."""
    expires = int(clock()) + 60
    signature = UrlSigner(SIGNING_SECRET).sign(uuid.UUID(job_id), 0, expires)
    return f"/images/{job_id}/0.png?expires={expires}&sig={signature}"
