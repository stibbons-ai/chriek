"""Property 4: one job at a time in persisted FIFO order; restarts fail running jobs only."""

from conftest import ALICE, BOB, GatedBackend, running, submit, wait_for


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
