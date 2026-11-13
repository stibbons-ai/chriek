"""Property 4: one job at a time in persisted FIFO order; restarts fail running jobs only."""

import io
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import (
    ALICE,
    BOB,
    CountingBackend,
    FailingBackend,
    GatedBackend,
    connect_db,
    db_dump,
    running,
    submit,
    tree_state,
    wait_for,
)
from PIL import Image
from sqlalchemy.exc import OperationalError

from chriek.api import WORKER_STOPPED
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


def test_restart_fails_running_job_and_runs_queued_job(tmp_path):
    crashed_backend = GatedBackend()
    with running(tmp_path, backend=crashed_backend) as before:
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

            # Let the stale process finish its render; stopping it waits for that.
            crashed_backend.release()

    # The stale process's late result must not resurrect the failed job.
    with running(tmp_path) as later:
        stale = later.get(f"/jobs/{running_job['id']}", headers=ALICE).json()
        assert stale["status"] == "failed"
        assert stale["images"] == []
    assert crashed_backend.prompts == ["interrupted"]
    assert not (tmp_path / "images" / running_job["id"]).exists()


def test_jobs_and_downloads_survive_a_clean_restart(tmp_path):
    with running(tmp_path) as before:
        body = wait_for(before, submit(before, width=16, height=16)["id"])
        assert body["status"] == "succeeded"

    with running(tmp_path) as after:
        again = after.get(f"/jobs/{body['id']}", headers=ALICE).json()
        assert again["status"] == "succeeded"
        assert again["params"] == body["params"]
        assert after.get(again["images"][0]["url"]).status_code == 200


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


@pytest.mark.parametrize("sentinel", [":memory:", "./:memory:"])
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
    # Arrange: the first final-state write raises, as if SQLite were locked or the disk full.
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

    with running(tmp_path, backend=backend) as before:
        first = submit(before, prompt="first", width=16, height=16)
        second = submit(before, prompt="second", width=16, height=16)
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

        # Act/assert: valid submissions from any owner are refused with nothing stored.
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
        stored = db.execute("SELECT id FROM jobs ORDER BY seq").fetchall()
    assert stored == [(first["id"],), (second["id"],)], "a rejected job was persisted"
