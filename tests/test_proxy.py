"""In proxy mode every client is one shared owner, whatever headers it sends."""

import uuid
from urllib.parse import urlsplit

import pytest
from conftest import (
    ALICE,
    ALICE_KEY,
    BOB,
    SMALL,
    StepBackend,
    post_job,
    proxy_env,
    running,
    submit,
    wait_for,
    with_query,
)

# What clients behind a proxy may send: nothing, keys valid or invalid in api-key mode,
# other schemes, and identity headers chriek must not trust.
CLIENT_HEADERS = [
    {},
    ALICE,
    BOB,
    {"Authorization": "Bearer not-a-key"},
    {"Authorization": f"Basic {ALICE_KEY}"},
    {"X-Forwarded-User": "carol"},
    {"X-Remote-User": "dave", "X-Forwarded-Email": "dave@example.com"},
]


def test_every_client_shares_one_owner_and_signed_downloads_still_apply(tmp_path):
    with running(tmp_path, env_base=proxy_env) as client:
        jobs = [submit(client, headers=headers, **SMALL) for headers in CLIENT_HEADERS]

        for job in jobs:
            for headers in CLIENT_HEADERS:
                body = wait_for(client, job["id"], headers=headers)
                assert body["status"] == "succeeded", body
        assert client.get(f"/jobs/{uuid.uuid7()}").status_code == 404

        url = wait_for(client, jobs[0]["id"])["images"][0]["url"]
        assert client.get(url).status_code == 200
        forged = with_query(url, sig="0" * 64)
        unsigned = urlsplit(url)._replace(query="").geturl()
        for response in (client.get(forged), client.get(unsigned)):
            assert (response.status_code, response.json()) == (403, {"detail": "invalid signature"})


def test_proxy_mode_never_answers_401(tmp_path):
    with running(tmp_path, env_base=proxy_env) as client:
        malformed = client.post("/jobs", headers={"Content-Type": "application/json"}, content=b"{")
        assert malformed.status_code == 422
        assert client.get("/jobs/not-a-uuid").status_code == 404


def test_proxy_clients_share_one_queue_capacity(tmp_path):
    backend = StepBackend()
    with running(
        tmp_path, backend=backend, env_base=proxy_env, CHRIEK_MAX_QUEUED_JOBS="2"
    ) as client:
        jobs = [submit(client, prompt="running", **SMALL)]
        backend.wait_started(1)
        jobs.append(submit(client, headers=ALICE, prompt="queued-1", **SMALL))
        jobs.append(
            submit(client, headers={"X-Forwarded-User": "carol"}, prompt="queued-2", **SMALL)
        )

        response = post_job(client, "rejected", headers=BOB, **SMALL)

        assert response.status_code == 429
        assert "at most 2 queued jobs per owner" in response.json()["detail"]
        backend.step(3)
        for job in jobs:
            assert wait_for(client, job["id"])["status"] == "succeeded"
    assert backend.prompts == ["running", "queued-1", "queued-2"]


@pytest.mark.parametrize("key_name", ["proxy", "shared", "proxy_shared"])
def test_jobs_never_cross_between_api_key_and_proxy_mode(tmp_path, key_name):
    with running(tmp_path, CHRIEK_API_KEYS=f"{key_name}:{ALICE_KEY}") as api_key_mode:
        keyed = wait_for(
            api_key_mode, submit(api_key_mode, headers=ALICE, **SMALL)["id"], headers=ALICE
        )

    with running(tmp_path, env_base=proxy_env) as proxy_mode:
        assert proxy_mode.get(f"/jobs/{keyed['id']}", headers=ALICE).status_code == 404
        proxied = wait_for(proxy_mode, submit(proxy_mode, **SMALL)["id"])

    with running(tmp_path, CHRIEK_API_KEYS=f"{key_name}:{ALICE_KEY}") as api_key_mode:
        assert api_key_mode.get(f"/jobs/{keyed['id']}", headers=ALICE).status_code == 200
        assert api_key_mode.get(f"/jobs/{proxied['id']}", headers=ALICE).status_code == 404


def test_explicit_api_key_mode_still_requires_a_key(tmp_path):
    with running(tmp_path, CHRIEK_AUTH_MODE="api-key") as client:
        job = submit(client, headers=ALICE, **SMALL)
        assert client.post("/jobs", json={"prompt": "x"}).status_code == 401
        assert (
            client.get(f"/jobs/{job['id']}", headers={"X-Forwarded-User": "alice"}).status_code
            == 401
        )
        assert client.get(f"/jobs/{job['id']}", headers=BOB).status_code == 404
