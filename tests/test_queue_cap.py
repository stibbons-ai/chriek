"""Admission never grows one owner's queued jobs beyond CHRIEK_MAX_QUEUED_JOBS."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

from conftest import ALICE, BOB, SMALL, StepBackend, post_job, running, submit, wait_for


def queue_full(cap: int) -> dict[str, str]:
    return {
        "detail": f"queue full: at most {cap} queued jobs per owner; "
        "retry after some of them have started"
    }


def post(client, prompt: str, headers=ALICE):
    return post_job(client, prompt, headers=headers, **SMALL)


def stored_prompts(tmp_path) -> list[str]:
    with sqlite3.connect(tmp_path / "db" / "chriek.db") as db:
        return [row[0] for row in db.execute("SELECT prompt FROM jobs ORDER BY seq")]


def test_full_queue_rejects_with_429_without_storing_or_running_the_job(tmp_path):
    backend = StepBackend()
    with running(tmp_path, backend=backend, CHRIEK_MAX_QUEUED_JOBS="2") as client:
        running_job = submit(client, prompt="running", **SMALL)
        backend.wait_started(1)  # a running job doesn't count against the cap
        queued = [submit(client, prompt=f"queued-{n}", **SMALL) for n in range(2)]

        rejected = post(client, "rejected")

        assert rejected.status_code == 429
        assert rejected.json() == queue_full(2)
        assert "location" not in rejected.headers
        invalid = client.post("/jobs", headers=ALICE, json={"prompt": "x", "width": 17})
        assert invalid.status_code == 422, "validation comes before the queue limit"
        assert stored_prompts(tmp_path) == ["running", "queued-0", "queued-1"]

        backend.step(3)
        for job in (running_job, *queued):
            assert wait_for(client, job["id"])["status"] == "succeeded"
    assert backend.prompts == ["running", "queued-0", "queued-1"]


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


def test_concurrent_submissions_through_two_servers_never_exceed_the_cap(tmp_path):
    # Two app instances on one database file: each has its own SQLite connections, like two
    # `chriek serve` processes. Both workers are kept busy so queued jobs stay queued.
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
