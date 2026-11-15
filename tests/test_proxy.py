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
    base_env,
    post_job,
    proxy_env,
    running,
    stored_owners,
    submit,
    wait_for,
    with_query,
)

from chriek.auth import PROXY_OWNER
from chriek.config import ConfigError, Settings

# What clients behind a proxy may send: nothing, keys valid or invalid in api-key mode,
# other schemes, and representative identity headers chriek must not trust. A sample,
# not every header a proxy could send.
CLIENT_HEADERS = [
    {},
    ALICE,
    BOB,
    {"Authorization": "Bearer not-a-key"},
    {"Authorization": f"Basic {ALICE_KEY}"},
    {"X-Forwarded-User": "carol"},
    {"X-Remote-User": "dave", "X-Forwarded-Email": "dave@example.com"},
    {"X-Authenticated-User": "erin"},
    {"Remote-User": "frank"},
    {"Cookie": "session=grace; user=grace"},
    {"X-Forwarded-Host": "evil.example"},  # the Host check reads Host only
    {"X-Forwarded-For": "203.0.113.7"},
    {"Forwarded": "for=203.0.113.8;proto=https;by=203.0.113.1"},
    {"X-Real-IP": "203.0.113.9"},
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
    assert stored_owners(tmp_path) == [PROXY_OWNER] * len(CLIENT_HEADERS)


@pytest.mark.parametrize(
    ("allowed", "host"),
    [
        ("testserver", "testserver"),
        ("testserver", "TestServer"),
        ("TESTSERVER", "testserver:8000"),
        ("img.example.com", "img.example.com:443"),
        ("127.0.0.1", "127.0.0.1:8000"),
        ("[::1]", "[::1]"),
        ("[::1]", "[::1]:8000"),
        ("localhost, [::1]", "[::1]"),
    ],
)
def test_proxy_mode_accepts_allowed_hosts(tmp_path, allowed, host):
    with running(tmp_path, env_base=proxy_env, CHRIEK_ALLOWED_HOSTS=allowed) as client:
        job = submit(client, headers={"Host": host}, **SMALL)
        body = wait_for(client, job["id"], headers={"Host": host})
        url = body["images"][0]["url"]
        # Links use CHRIEK_PUBLIC_URL, never the accepted Host.
        assert url.startswith(f"http://testserver/images/{job['id']}/0.png?"), url
        assert client.get(url, headers={"Host": host}).status_code == 200


HOSTILE_HOSTS = [
    "evil.example",
    "evil.example:8000",
    "testserver.evil.example",
    "www.testserver",
    "localhost",  # a default entry, replaced by the configured list
    "127.0.0.1",
    "[::2]",
    "[::1",
    "testserver:port",
    "testserver/path",
    "test server",
    "",
]


def test_proxy_mode_rejects_other_hosts_before_any_job_work(tmp_path):
    with running(tmp_path, env_base=proxy_env) as client:
        job = submit(client, **SMALL)
        url = wait_for(client, job["id"])["images"][0]["url"]

        for host in HOSTILE_HOSTS:
            hostile = {"Host": host, "X-Forwarded-Host": "testserver"}
            for response in (
                post_job(client, "rebound", headers=hostile, **SMALL),
                client.get(f"/jobs/{job['id']}", headers=hostile),
                client.get(url, headers=hostile),
                client.get("/health", headers=hostile),
            ):
                assert (response.status_code, response.text) == (400, "Invalid host header"), host
    assert stored_owners(tmp_path) == [PROXY_OWNER]


def test_proxy_mode_rejects_the_bare_name_of_an_allowed_www_host(tmp_path):
    # Starlette would otherwise redirect img.example.com to the allowed www. name.
    allowed = {"CHRIEK_ALLOWED_HOSTS": "www.img.example.com"}
    with running(tmp_path, env_base=proxy_env, **allowed) as client:
        response = client.post(
            "/jobs",
            headers={"Host": "img.example.com"},
            json={"prompt": "rebound"} | SMALL,
            follow_redirects=False,
        )
    assert (response.status_code, response.text) == (400, "Invalid host header")
    assert "location" not in response.headers
    assert stored_owners(tmp_path) == []


@pytest.mark.parametrize(
    "hosts",
    [
        "",
        " ",
        "localhost,,img.example.com",
        "*",
        "*.example.com",
        "img.example.com:443",
        "https://img.example.com",
        "img.example.com/jobs",
        "::1",
        "[::1]:8000",
        "[fe80::1%eth0]",
        "[not-ipv6]",
        "-img.example.com",
    ],
)
def test_allowed_hosts_must_be_exact_names(tmp_path, hosts):
    with pytest.raises(ConfigError, match="CHRIEK_ALLOWED_HOSTS entry"):
        Settings.from_env(proxy_env(tmp_path) | {"CHRIEK_ALLOWED_HOSTS": hosts})


def test_api_key_mode_does_not_check_host(tmp_path):
    with running(tmp_path) as client:
        job = submit(client, headers=ALICE | {"Host": "evil.example"}, **SMALL)
        assert wait_for(client, job["id"], headers=ALICE | {"Host": "evil.example"})["images"]


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


def test_no_api_key_name_can_own_the_proxy_owners_jobs(tmp_path):
    # Key names can't contain ":", so configuring the proxy owner as a key name is either
    # rejected or parsed as a different legal name: no api-key owner is the proxy owner.
    env = base_env(tmp_path) | {"CHRIEK_API_KEYS": f"{PROXY_OWNER}:{ALICE_KEY}"}
    try:
        owners = Settings.from_env(env).api_keys.keys()
    except ConfigError:
        owners = set()
    assert PROXY_OWNER not in owners


def test_explicit_api_key_mode_still_requires_a_key(tmp_path):
    with running(tmp_path, CHRIEK_AUTH_MODE="api-key") as client:
        job = submit(client, headers=ALICE, **SMALL)
        assert client.post("/jobs", json={"prompt": "x"}).status_code == 401
        assert (
            client.get(f"/jobs/{job['id']}", headers={"X-Forwarded-User": "alice"}).status_code
            == 401
        )
        assert client.get(f"/jobs/{job['id']}", headers=BOB).status_code == 404
