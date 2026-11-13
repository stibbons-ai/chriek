"""Property 1: jobs are visible only to the API-key owner that created them."""

import uuid

import pytest
from conftest import (
    ALICE,
    ALICE_KEY,
    BOB,
    CountingBackend,
    assert_nothing_queued,
    running,
    submit,
    wait_for,
)

VALID_JOB = {"prompt": "a lighthouse"}


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": ""},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer "},
        {"Authorization": f"Basic {ALICE_KEY}"},
        {"Authorization": ALICE_KEY},
        {"Authorization": f"Bearer {ALICE_KEY}x"},
        {"Authorization": f"Bearer {ALICE_KEY[:-1]}"},
        {"Authorization": "Bearer alice"},
        {"X-API-Key": ALICE_KEY},
    ],
    ids=[
        "missing",
        "empty",
        "scheme-only",
        "blank-key",
        "basic-scheme",
        "no-scheme",
        "key-plus-suffix",
        "key-prefix",
        "owner-name-as-key",
        "other-header",
    ],
)
def test_missing_or_invalid_key_is_401_on_every_job_endpoint(client, headers):
    job = submit(client)

    for response in (
        client.post("/jobs", headers=headers, json=VALID_JOB),
        client.get(f"/jobs/{job['id']}", headers=headers),
        client.get(f"/jobs/{uuid.uuid7()}", headers=headers),
    ):
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json() == {"detail": "missing or invalid API key"}


def test_bearer_scheme_is_case_insensitive(client):
    response = client.post(
        "/jobs", headers={"Authorization": f"bearer {ALICE_KEY}"}, json=VALID_JOB
    )
    assert response.status_code == 202


def test_invalid_body_without_key_is_401_not_422(client):
    assert client.post("/jobs", json={"width": "nope"}).status_code == 401


@pytest.mark.parametrize(
    "headers", [{}, {"Authorization": "Bearer invalid"}], ids=["missing", "invalid"]
)
@pytest.mark.parametrize(
    "content", [b"{", b'{"prompt": "bad\xff"}'], ids=["malformed-json", "invalid-utf8"]
)
def test_unparseable_body_without_valid_key_is_401_and_never_executes(tmp_path, headers, content):
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        response = client.post(
            "/jobs", headers=headers | {"Content-Type": "application/json"}, content=content
        )

        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json() == {"detail": "missing or invalid API key"}
        assert_nothing_queued(client, backend)


def test_any_printable_ascii_key_authenticates(tmp_path):
    # Every printable ASCII character except the entry separator ",".
    key = "".join(chr(code) for code in range(ord("!"), ord("~") + 1) if chr(code) != ",")
    carol = {"Authorization": f"Bearer {key}"}
    with running(tmp_path, CHRIEK_API_KEYS=f"carol:{key}") as client:
        job = submit(client, headers=carol)
        assert client.get(f"/jobs/{job['id']}", headers=carol).status_code == 200


def test_other_owners_job_is_indistinguishable_from_a_missing_job(client):
    job = submit(client, headers=ALICE)
    finished = wait_for(client, job["id"], headers=ALICE)
    assert finished["status"] == "succeeded"

    other_owner = client.get(f"/jobs/{job['id']}", headers=BOB)
    missing = client.get(f"/jobs/{uuid.uuid7()}", headers=BOB)
    malformed = client.get("/jobs/not-a-uuid", headers=BOB)

    for response in (other_owner, missing, malformed):
        assert response.status_code == 404
        assert response.json() == {"detail": "job not found"}
    assert client.get(f"/jobs/{job['id']}", headers=ALICE).status_code == 200


def test_each_owner_sees_only_their_own_jobs(client):
    alice_job = submit(client, headers=ALICE, prompt="alice's prompt")
    bob_job = submit(client, headers=BOB, prompt="bob's prompt")

    assert wait_for(client, alice_job["id"], headers=ALICE)["params"]["prompt"] == "alice's prompt"
    assert wait_for(client, bob_job["id"], headers=BOB)["params"]["prompt"] == "bob's prompt"
    assert client.get(f"/jobs/{alice_job['id']}", headers=BOB).status_code == 404
    assert client.get(f"/jobs/{bob_job['id']}", headers=ALICE).status_code == 404


def test_health_needs_no_key_and_reveals_nothing(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
