"""Admission never grows one owner's queued jobs beyond CHRIEK_MAX_QUEUED_JOBS."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest
from conftest import ALICE, BOB, SMALL, StepBackend, base_env, post_job, running, submit, wait_for

from chriek.store import JobStore


def queue_full(cap: int) -> dict[str, str]:
    return {
        "detail": f"queue full: at most {cap} queued jobs per owner; "
        "new jobs are accepted only while fewer are queued"
    }


def post(client, prompt: str, headers=ALICE):
    return post_job(client, prompt, headers=headers, **SMALL)


def db_file(tmp_path) -> str:
    return base_env(tmp_path)["CHRIEK_DB_PATH"]


def stored_prompts(tmp_path) -> list[str]:
    with sqlite3.connect(db_file(tmp_path)) as db:
        return [row[0] for row in db.execute("SELECT prompt FROM jobs ORDER BY seq")]


@pytest.mark.parametrize(
    ("env", "cap"),
    [
        ({"CHRIEK_MAX_QUEUED_JOBS": "2"}, 2),
        ({}, 100),
        ({"CHRIEK_MAX_QUEUED_JOBS": "2", "CHRIEK_DB_PATH": ":memory:"}, 2),
    ],
    ids=["configured", "default", "memory-db"],
)
def test_full_queue_rejects_with_429_without_storing_or_running_the_job(tmp_path, env, cap):
    backend = StepBackend()
    with running(tmp_path, backend=backend, **env) as client:
        running_job = submit(client, prompt="running", **SMALL)
        backend.wait_started(1)  # a running job doesn't count against the cap
        queued = [submit(client, prompt=f"queued-{n}", **SMALL) for n in range(cap)]

        rejected = post(client, "rejected")

        assert rejected.status_code == 429
        assert rejected.json() == queue_full(cap)
        assert "location" not in rejected.headers
        invalid = client.post("/jobs", headers=ALICE, json={"prompt": "x", "width": 17})
        assert invalid.status_code == 422, "validation comes before the queue limit"

        backend.step(cap + 2)
        for job in (running_job, *queued):
            assert wait_for(client, job["id"])["status"] == "succeeded"
        # FIFO: had the rejected job been stored, it would have run before this one.
        after = submit(client, prompt="after", **SMALL)
        assert wait_for(client, after["id"])["status"] == "succeeded"
    assert backend.prompts == ["running", *(f"queued-{n}" for n in range(cap)), "after"]


def test_full_queue_rejects_while_another_connection_holds_the_write_lock(tmp_path):
    backend = StepBackend()
    with running(tmp_path, backend=backend, CHRIEK_MAX_QUEUED_JOBS="1") as client:
        held = submit(client, prompt="running", **SMALL)
        backend.wait_started(1)
        queued = submit(client, prompt="queued", **SMALL)

        # Another connection holds the write lock across these requests. A rejection that
        # needed the lock would block for SQLite's busy timeout and then raise "database is
        # locked" (a 500 in production; TestClient re-raises it).
        writer = sqlite3.connect(db_file(tmp_path), timeout=0, isolation_level=None)
        try:
            writer.execute("BEGIN IMMEDIATE")
            for _ in range(3):
                rejected = post(client, "rejected")
                assert (rejected.status_code, rejected.json()) == (429, queue_full(1))
            assert client.get(f"/jobs/{queued['id']}", headers=ALICE).status_code == 200
        finally:
            writer.close()

        backend.step(2)
        for job in (held, queued):
            assert wait_for(client, job["id"])["status"] == "succeeded"
    assert stored_prompts(tmp_path) == ["running", "queued"]


def test_each_owner_has_an_independent_queue(tmp_path):
    backend = StepBackend()
    with running(tmp_path, backend=backend, CHRIEK_MAX_QUEUED_JOBS="1") as client:
        submit(client, prompt="alice-running", **SMALL)
        backend.wait_started(1)
        submit(client, prompt="alice-queued", **SMALL)
        assert post(client, "alice-over").status_code == 429

        bob_queued = post(client, "bob-queued", headers=BOB)
        bob_over = post(client, "bob-over", headers=BOB)

        assert bob_queued.status_code == 202
        assert (bob_over.status_code, bob_over.json()) == (429, queue_full(1))
        assert post(client, "alice-still-over").status_code == 429
        backend.step(3)
        assert wait_for(client, bob_queued.json()["id"], headers=BOB)["status"] == "succeeded"


def test_capacity_returns_once_a_queued_job_starts_running(tmp_path):
    backend = StepBackend()
    with running(tmp_path, backend=backend, CHRIEK_MAX_QUEUED_JOBS="1") as client:
        first = submit(client, prompt="first", **SMALL)
        backend.wait_started(1)
        second = submit(client, prompt="second", **SMALL)
        assert post(client, "too-early").status_code == 429

        backend.step()  # first finishes; the worker moves second from queued to running
        backend.wait_started(2)
        assert client.get(f"/jobs/{second['id']}", headers=ALICE).json()["status"] == "running"
        third = post(client, "third")
        assert third.status_code == 202, third.text
        assert post(client, "fourth").status_code == 429

        backend.step(2)
        for job in (first, second, third.json()):
            assert wait_for(client, job["id"])["status"] == "succeeded"
        # Finished jobs don't count either.
        assert post(client, "after").status_code == 202
        backend.wait_started(4)
        backend.step()
    assert backend.prompts == ["first", "second", "third", "after"]


class FailingStepBackend(StepBackend):
    """StepBackend whose render of the prompt "fails" raises once stepped."""

    def render(self, prompt, width, height, seed):
        image = super().render(prompt, width, height, seed)
        if prompt == "fails":
            raise RuntimeError("injected backend failure")
        return image


def test_failed_jobs_do_not_take_queue_capacity(tmp_path):
    backend = FailingStepBackend()
    with running(tmp_path, backend=backend, CHRIEK_MAX_QUEUED_JOBS="1") as client:
        failed = submit(client, prompt="fails", **SMALL)
        backend.step()
        assert wait_for(client, failed["id"])["status"] == "failed"
        held = submit(client, prompt="running", **SMALL)
        backend.wait_started(2)

        queued = post(client, "queued")
        assert queued.status_code == 202, "a failed job must not use the only queued slot"
        over = post(client, "over")
        assert (over.status_code, over.json()) == (429, queue_full(1))

        backend.step(2)
        for job in (held, queued.json()):
            assert wait_for(client, job["id"])["status"] == "succeeded"
    assert backend.prompts == ["fails", "running", "queued"]


def test_concurrent_submissions_through_separate_connections_never_exceed_the_cap(tmp_path):
    # Two app instances on one database file only to get independent SQLite connections
    # (one server per database is supported). Both workers stay busy so jobs stay queued.
    backend = StepBackend()
    cap, attempts = 3, 16
    env = {"CHRIEK_MAX_QUEUED_JOBS": str(cap)}
    with (
        running(tmp_path, backend=backend, **env) as one,
        running(tmp_path, backend=backend, **env) as two,
    ):
        busy = [submit(one, headers=BOB, prompt="busy-1", **SMALL)]
        backend.wait_started(1)
        busy.append(submit(two, headers=BOB, prompt="busy-2", **SMALL))
        backend.wait_started(2)

        start = threading.Barrier(attempts)

        def race(n: int):
            client = (one, two)[n % 2]
            start.wait(10)
            return post(client, f"race-{n}")

        with ThreadPoolExecutor(attempts) as pool:
            responses = list(pool.map(race, range(attempts)))

        statuses = sorted(response.status_code for response in responses)
        assert statuses == [202] * cap + [429] * (attempts - cap)
        admitted = [r.json() for r in responses if r.status_code == 202]
        assert sorted(p for p in stored_prompts(tmp_path) if p.startswith("race-")) == sorted(
            job["params"]["prompt"] for job in admitted
        )
        assert post(two, "bob-has-room", headers=BOB).status_code == 202

        backend.step(len(busy) + cap + 1)
        for job in admitted:
            assert wait_for(one, job["id"])["status"] == "succeeded"


def test_existing_database_gains_the_owner_queue_index_and_keeps_its_jobs(tmp_path):
    # A database file from before the owner/status index: same schema, no such index.
    path = tmp_path / "existing.db"
    store = JobStore(path)
    job = {"prompt": "p", "image_count": 2, "width": 16, "height": 16, "seed": 7}
    for n in range(4):
        store.create(owner="alice", now=float(n), max_queued=100, **job)
    succeeded, failed, _running = (store.claim_next(now=10.0) for _ in range(3))
    store.succeed(succeeded.id, {"backend": "fake"}, now=11.0)
    store.fail(failed.id, "boom", {"backend": "fake"}, now=12.0)
    store.close()

    def owner_status_indexes(db) -> list[str]:
        return [
            name
            for _, name, *_ in db.execute("PRAGMA index_list(jobs)")
            if [row[2] for row in db.execute(f"PRAGMA index_info({name})")] == ["owner", "status"]
        ]

    def snapshot() -> tuple[list, list, list[str]]:
        with closing(sqlite3.connect(path)) as db:
            jobs = db.execute("SELECT * FROM jobs ORDER BY seq").fetchall()
            images = db.execute('SELECT * FROM images ORDER BY job_id, "index"').fetchall()
            return jobs, images, owner_status_indexes(db)

    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        for name in owner_status_indexes(db):
            db.execute(f"DROP INDEX {name}")
        statuses = [row[0] for row in db.execute("SELECT status FROM jobs ORDER BY seq")]
    assert statuses == ["succeeded", "failed", "running", "queued"]
    jobs, images, indexes = snapshot()
    assert len(images) == 2
    assert indexes == []

    for _ in range(2):  # starting again on an upgraded file changes nothing
        store = JobStore(path)
        try:
            upgraded_jobs, upgraded_images, indexes = snapshot()
            assert (upgraded_jobs, upgraded_images) == (jobs, images)
            assert len(indexes) == 1
            assert store.is_published(succeeded.id, 1)
        finally:
            store.close()

    store = JobStore(path)
    try:
        assert store.create(owner="alice", now=20.0, max_queued=1, **job) is None
        assert store.create(owner="alice", now=21.0, max_queued=2, **job) is not None
    finally:
        store.close()
