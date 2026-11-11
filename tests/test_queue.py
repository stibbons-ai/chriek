"""Property 4: one job at a time in persisted FIFO order; restarts fail running jobs only."""

import logging
import time

import pytest
from conftest import (
    ALICE,
    BOB,
    CountingBackend,
    FailingBackend,
    GatedBackend,
    running,
    submit,
    wait_for,
)
from sqlalchemy.exc import OperationalError

from chriek.store import JobStore


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


@pytest.mark.parametrize(
    ("store_method", "first_render_fails"),
    [("succeed", False), ("fail", True)],
    ids=["success-write-fails", "failure-write-fails"],
)
def test_unsaved_job_result_stops_the_worker_until_restart(
    tmp_path, monkeypatch, caplog, store_method, first_render_fails
):
    # Arrange: the first final-state write raises, as if SQLite were locked or the disk full.
    original = getattr(JobStore, store_method)
    calls = []

    def broken_once(self, *args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise OperationalError("COMMIT", {}, RuntimeError("database is locked"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(JobStore, store_method, broken_once)
    backend = FailingBackend(fail_index=0) if first_render_fails else CountingBackend()

    with running(tmp_path, backend=backend) as before:
        first = submit(before, prompt="first", width=16, height=16)
        second = submit(before, prompt="second", width=16, height=16)

        deadline = time.monotonic() + 10
        while (health := before.get("/health")).status_code != 503:
            assert time.monotonic() < deadline, "worker never reported its failure"
            time.sleep(0.01)
        assert health.json() == {
            "status": "error",
            "detail": "worker stopped: a job result could not be saved; restart chriek",
        }
        stuck = before.get(f"/jobs/{first['id']}", headers=ALICE).json()
        assert (stuck["status"], stuck["images"], stuck["error"]) == ("running", [], None)
        assert status(before, second["id"]) == "queued"
        third = submit(before, prompt="third", width=16, height=16)
        assert status(before, third["id"]) == "queued"
        assert backend.renders == 1, "no job may run after an unsaved result"

    assert any(
        isinstance(record.exc_info[1], OperationalError)
        for record in caplog.records
        if record.levelno == logging.ERROR and record.exc_info
    ), "the original database error must be logged"

    with running(tmp_path) as after:
        assert after.get("/health").json() == {"status": "ok"}
        interrupted = after.get(f"/jobs/{first['id']}", headers=ALICE).json()
        assert interrupted["status"] == "failed"
        assert interrupted["error"] == "interrupted: the server stopped while this job was running"
        assert (interrupted["images"], interrupted["run"]) == ([], None)
        assert wait_for(after, second["id"])["status"] == "succeeded"
        assert wait_for(after, third["id"])["status"] == "succeeded"
