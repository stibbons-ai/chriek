"""Serverless roles: `chriek api` dispatches each job ID over HTTP to `chriek worker`.

Configuration cases run everywhere. The walkthroughs need the shared Postgres database and S3
bucket the roles require, so they run only with both CHRIEK_TEST_POSTGRES_URL and
CHRIEK_TEST_S3_URL (see conftest.py); the worker serves real HTTP on a loopback port.
"""

import io
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import uvicorn
from conftest import (
    ALICE,
    ALICE_KEY,
    POSTGRES_URL,
    S3_URL,
    SIGNING_SECRET,
    SMALL,
    CountingBackend,
    FailingBackend,
    GatedBackend,
    base_env,
    fetch,
    make_image_bytes,
    submit,
    wait_for,
)
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import make_url
from test_references import submit_multipart

from chriek import api as api_module
from chriek.api import create_app, create_worker_app
from chriek.cli import main
from chriek.config import (
    WORKER_SETTING_NAMES,
    ConfigError,
    Role,
    Settings,
    WorkerSettings,
)
from chriek.store import JobStore

WORKER_TOKEN = "worker-token-0123456789abcdefghijklmnopq"
PROVIDER_TOKEN = "provider-token-SENTINEL-0123456789"
needs_services = pytest.mark.skipif(
    POSTGRES_URL is None or S3_URL is None,
    reason="serverless roles need CHRIEK_TEST_POSTGRES_URL and CHRIEK_TEST_S3_URL",
)
SHARED = {  # configuration-only stand-ins for the shared database and bucket
    "CHRIEK_DATABASE_URL": "postgresql://chriek@db.invalid/chriek",
    "CHRIEK_STORAGE": "s3",
    "CHRIEK_S3_BUCKET": "chriek-images",
    "CHRIEK_S3_ACCESS_KEY_ID": "AKIDEXAMPLE",
    "CHRIEK_S3_SECRET_ACCESS_KEY": "s3-secret-0123456789",
}


def api_env(tmp_path: Path, url: str, **changes: str) -> dict[str, str]:
    return (
        base_env(tmp_path)
        | {
            "CHRIEK_DISPATCH_URL": url,
            "CHRIEK_DISPATCH_TOKEN": PROVIDER_TOKEN,
            "CHRIEK_DISPATCH_HEADERS": json.dumps({"Prefer": "respond-async"}),
            "CHRIEK_WORKER_TOKEN": WORKER_TOKEN,
        }
        | changes
    )


def worker_env(tmp_path: Path) -> dict[str, str]:
    env = {k: v for k, v in base_env(tmp_path).items() if k in WORKER_SETTING_NAMES}
    return env | {"CHRIEK_WORKER_TOKEN": WORKER_TOKEN}


def config_api_env(tmp_path: Path, **changes: str) -> dict[str, str]:
    env = (
        base_env(tmp_path)
        | SHARED
        | {
            "CHRIEK_DISPATCH_URL": "https://worker.example/invoke",
            "CHRIEK_WORKER_TOKEN": WORKER_TOKEN,
        }
    )
    env.pop("CHRIEK_STORAGE_DIR", None)
    return env | changes


# --- configuration -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "removed", "message"),
    [
        ({}, "CHRIEK_DATABASE_URL", "chriek api requires CHRIEK_DATABASE_URL"),
        (
            {"CHRIEK_STORAGE": "local"},
            "CHRIEK_S3_BUCKET",
            "requires CHRIEK_STORAGE=s3",
        ),
        ({}, "CHRIEK_DISPATCH_URL", "CHRIEK_DISPATCH_URL is required"),
        ({}, "CHRIEK_WORKER_TOKEN", "CHRIEK_WORKER_TOKEN must be at least 32"),
        ({"CHRIEK_WORKER_TOKEN": ALICE_KEY}, None, "CHRIEK_WORKER_TOKEN must differ"),
        ({"CHRIEK_WORKER_TOKEN": SIGNING_SECRET}, None, "CHRIEK_WORKER_TOKEN must differ"),
        ({"CHRIEK_BACKEND": "fake"}, None, "CHRIEK_BACKEND does not apply to chriek api"),
        ({"CHRIEK_DISPATCH_URL": "ftp://worker.example"}, None, "CHRIEK_DISPATCH_URL must be"),
        ({"CHRIEK_DISPATCH_HEADERS": "Prefer: respond-async"}, None, "must be a JSON object"),
        ({"CHRIEK_DISPATCH_HEADERS": '{"Prefer": 1}'}, None, "must be a JSON object"),
        (
            {"CHRIEK_DISPATCH_HEADERS": json.dumps({"authorization": PROVIDER_TOKEN})},
            None,
            "must not set Authorization",
        ),
        (
            {"CHRIEK_DISPATCH_HEADERS": json.dumps({"X-Key": f"{PROVIDER_TOKEN}\r\nX: y"})},
            None,
            "values printable ASCII",
        ),
        ({"CHRIEK_DISPATCH_TOKEN": f"{PROVIDER_TOKEN} x"}, None, "CHRIEK_DISPATCH_TOKEN must"),
    ],
)
def test_api_role_rejects_configuration_without_leaking_secrets(
    tmp_path, changes, removed, message
):
    env = config_api_env(tmp_path, **changes)
    if removed:
        env.pop(removed)
        if removed == "CHRIEK_S3_BUCKET":
            for name in [n for n in env if n.startswith("CHRIEK_S3_")]:
                del env[name]
    with pytest.raises(ConfigError) as error:
        Settings.from_env(env, Role.API)
    assert message in str(error.value)
    for secret in (PROVIDER_TOKEN, WORKER_TOKEN, ALICE_KEY, SIGNING_SECRET):
        assert secret not in str(error.value)


def test_api_role_settings_carry_dispatch_headers(tmp_path):
    settings = Settings.from_env(
        config_api_env(
            tmp_path,
            CHRIEK_DISPATCH_TOKEN=PROVIDER_TOKEN,
            CHRIEK_DISPATCH_HEADERS=json.dumps({"Prefer": "respond-async"}),
        ),
        Role.API,
    )
    assert settings.dispatch.headers == {
        "Prefer": "respond-async",
        "Authorization": f"Bearer {PROVIDER_TOKEN}",
    }
    assert PROVIDER_TOKEN not in repr(settings) and WORKER_TOKEN not in repr(settings)


@pytest.mark.parametrize(
    "name", ["CHRIEK_DISPATCH_URL", "CHRIEK_DISPATCH_TOKEN", "CHRIEK_WORKER_TOKEN"]
)
def test_home_role_refuses_serverless_settings(tmp_path, name):
    with pytest.raises(ConfigError, match=f"{name} does not apply to chriek serve"):
        Settings.from_env(base_env(tmp_path) | {name: WORKER_TOKEN})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"CHRIEK_API_KEYS": f"alice:{ALICE_KEY}"}, "CHRIEK_API_KEYS does not apply"),
        ({"CHRIEK_SIGNING_SECRET": SIGNING_SECRET}, "CHRIEK_SIGNING_SECRET does not apply"),
        ({"CHRIEK_DISPATCH_URL": "https://x.example"}, "CHRIEK_DISPATCH_URL does not apply"),
        ({"CHRIEK_DB_PATH": "chriek.db"}, "CHRIEK_DB_PATH does not apply"),
        ({"CHRIEK_WORKER_TOKEN": "short"}, "CHRIEK_WORKER_TOKEN must be at least 32"),
        ({"CHRIEK_DATABASE_URL": None}, "chriek worker requires CHRIEK_DATABASE_URL"),
        ({"CHRIEK_STORAGE": None, "CHRIEK_S3_BUCKET": None}, "requires CHRIEK_STORAGE=s3"),
    ],
)
def test_worker_role_rejects_configuration(changes, message):
    env = SHARED | {"CHRIEK_WORKER_TOKEN": WORKER_TOKEN} | changes
    env = {name: value for name, value in env.items() if value is not None}
    if "CHRIEK_STORAGE" not in env:
        env = {name: value for name, value in env.items() if not name.startswith("CHRIEK_S3_")}
    with pytest.raises(ConfigError, match=message) as error:
        WorkerSettings.from_env(env)
    assert SIGNING_SECRET not in str(error.value) and ALICE_KEY not in str(error.value)


@pytest.mark.parametrize("command", ["api", "worker"])
def test_role_commands_stop_on_sqlite_before_opening_anything(
    tmp_path, monkeypatch, capsys, command
):
    for name in [name for name in os.environ if name.upper().startswith("CHRIEK_")]:
        monkeypatch.delenv(name)
    env = {"CHRIEK_WORKER_TOKEN": WORKER_TOKEN}  # no CHRIEK_DATABASE_URL: SQLite
    if command == "api":
        env |= base_env(tmp_path) | {"CHRIEK_DISPATCH_URL": "https://worker.example/invoke"}
        env.pop("CHRIEK_DATABASE_URL", None)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: pytest.fail("server must not start"))

    assert main([command]) == 2
    error = capsys.readouterr().err
    assert error.startswith(
        f"chriek: configuration error: chriek {command} requires CHRIEK_DATABASE_URL"
    )
    assert WORKER_TOKEN not in error
    assert not (tmp_path / "db").exists()


# --- walkthroughs over HTTP ----------------------------------------------------------------


@contextmanager
def serving(app) -> Iterator[str]:
    """Serve `app` with uvicorn on a free loopback port in a thread; yield its base URL."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "server never started"
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()


class Provider:
    """A stand-in provider gateway: records each delivery, then optionally forwards it to the
    worker, and answers with `status`."""

    def __init__(self, status: int = 202, forward_to: str | None = None) -> None:
        self.deliveries: list[tuple[dict[str, str], dict]] = []
        self.app = FastAPI()

        @self.app.post("/{path:path}")
        def deliver(request_headers: Request, body: dict) -> Response:
            headers = dict(request_headers.headers)
            self.deliveries.append((headers, body))
            if forward_to:
                invoke(forward_to, body["job_id"])
            return Response(status_code=status)


def invoke(base: str, job_id: str, token: str = WORKER_TOKEN) -> tuple[int, dict]:
    request = urllib.request.Request(  # noqa: S310 - loopback test server
        f"{base}/invoke",
        data=json.dumps({"job_id": job_id}).encode(),
        headers={"Content-Type": "application/json", "X-Chriek-Worker-Token": token},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


@contextmanager
def worker(tmp_path: Path, backend) -> Iterator[str]:
    app = create_worker_app(WorkerSettings.from_env(worker_env(tmp_path)), backend=backend)
    with serving(app) as base:
        yield base


@contextmanager
def api(tmp_path: Path, url: str, **changes: str) -> Iterator[TestClient]:
    app = create_app(Settings.from_env(api_env(tmp_path, url, **changes), Role.API))
    with TestClient(app) as client:
        yield client


class ReferenceBackend(CountingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[tuple[int, ...]] = []

    def render(self, prompt, width, height, seed, references=()):
        self.references.append(tuple(len(ref.tobytes()) for ref in references))
        return super().render(prompt, width, height, seed, references)


def run_job(tmp_path: Path, url: str) -> dict:
    """Submit one small job to an API dispatching to `url`; return it once terminal."""
    with api(tmp_path, url) as client:
        return wait_for(client, submit(client, **SMALL)["id"])


@needs_services
def test_walkthrough_submit_dispatch_poll_download_with_reference(tmp_path, monkeypatch):
    # The API never builds a model: any attempt fails the test.
    monkeypatch.setattr(api_module, "create_backend", lambda *_: pytest.fail("API built a model"))
    backend = ReferenceBackend()
    with worker(tmp_path, backend) as base:
        provider = Provider(status=200, forward_to=base)
        with serving(provider.app) as provider_url, api(tmp_path, f"{provider_url}/run") as client:
            submitted = submit_multipart(
                client,
                data={"prompt": "a lighthouse", "image_count": "2", "width": "16", "height": "16"},
                files=[("references", ("r.png", make_image_bytes((200, 10, 10)), "image/png"))],
            )
            assert submitted["status"] == "queued"
            job = wait_for(client, submitted["id"])

    assert job["status"] == "succeeded", job
    assert job["params"]["reference_count"] == 1
    assert backend.renders == 2
    assert backend.references == [(16 * 16 * 3,), (16 * 16 * 3,)]
    for image in job["images"]:
        status, png = fetch(image["url"])
        assert status == 200
        assert Image.open(io.BytesIO(png)).size == (16, 16)
    ((headers, body),) = provider.deliveries
    assert body == {"job_id": job["id"]}
    assert headers["idempotency-key"] == job["id"]
    assert headers["prefer"] == "respond-async"
    assert headers["authorization"] == f"Bearer {PROVIDER_TOKEN}"
    assert headers["x-chriek-worker-token"] == WORKER_TOKEN


@needs_services
def test_duplicate_deliveries_render_once(tmp_path):
    backend = CountingBackend()
    provider = Provider(status=202)  # accepts and holds deliveries, like an async queue
    with (
        serving(provider.app) as provider_url,
        api(tmp_path, provider_url) as client,
        worker(tmp_path, backend) as base,
    ):
        job_id = submit(client, **SMALL)["id"]
        assert wait_for(client, job_id, statuses={"queued"})["status"] == "queued"

        assert invoke(base, job_id) == (200, {"id": job_id, "status": "succeeded"})
        assert invoke(base, job_id) == (200, {"id": job_id, "status": "succeeded"})
        assert backend.renders == 1
        assert wait_for(client, job_id)["status"] == "succeeded"


@needs_services
def test_concurrent_deliveries_to_two_workers_claim_once(tmp_path):
    backend = GatedBackend()
    provider = Provider(status=202)
    with (
        serving(provider.app) as provider_url,
        api(tmp_path, provider_url) as client,
        worker(tmp_path, backend) as first,
        worker(tmp_path, backend) as second,
    ):
        job_id = submit(client, **SMALL)["id"]
        results = []
        thread = threading.Thread(target=lambda: results.append(invoke(first, job_id)))
        thread.start()
        assert backend.rendering.wait(10)

        assert invoke(second, job_id) == (200, {"id": job_id, "status": "running"})
        # A restarted API recovers nothing: the running job is a worker's, not its own.
        with api(tmp_path, provider_url) as restarted:
            assert restarted.get(f"/jobs/{job_id}", headers=ALICE).json()["status"] == "running"
        backend.release()
        thread.join(20)
        assert wait_for(client, job_id)["status"] == "succeeded"

    assert results == [(200, {"id": job_id, "status": "succeeded"})]
    assert backend.prompts == ["a lighthouse"]


@needs_services
def test_render_failure_is_recorded_and_returned(tmp_path):
    with worker(tmp_path, FailingBackend(0)) as base, api(tmp_path, f"{base}/invoke") as client:
        job_id = submit(client, **SMALL)["id"]
        job = wait_for(client, job_id)
    assert job["status"] == "failed"
    assert job["error"] == "image 1 of 1 failed"
    assert job["images"] == []


@needs_services
@pytest.mark.parametrize("answer", [503, 307, None])
def test_delivery_failure_fails_the_queued_job_without_secrets(tmp_path, answer):
    secret_path = "/hooks/path-token-SENTINEL/invoke"
    if answer is None:  # nothing listens on a port just released
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        job = run_job(tmp_path, f"http://127.0.0.1:{port}{secret_path}")
    else:
        with serving(Provider(status=answer).app) as provider_url:
            job = run_job(tmp_path, f"{provider_url}{secret_path}")

    assert job["status"] == "failed"
    expected = (
        f"dispatch failed: the worker endpoint answered HTTP {answer}"
        if answer
        else "dispatch failed: the worker endpoint could not be reached (URLError)"
    )
    assert job["error"] == expected
    for secret in ("SENTINEL", PROVIDER_TOKEN, WORKER_TOKEN):
        assert secret not in json.dumps(job)


@needs_services
def test_delivery_failure_after_the_worker_ran_keeps_its_result(tmp_path):
    """A gateway that delivered the job but then answers 502 can't fail a finished job."""
    backend = CountingBackend()
    with worker(tmp_path, backend) as base:
        provider = Provider(status=502, forward_to=base)
        with serving(provider.app) as provider_url:
            job = run_job(tmp_path, provider_url)
    assert job["status"] == "succeeded"
    assert job["error"] is None
    assert backend.renders == 1


@needs_services
def test_worker_invocation_requires_its_token_and_health_does_not(tmp_path):
    backend = CountingBackend()
    provider = Provider(status=202)
    with (
        serving(provider.app) as provider_url,
        api(tmp_path, provider_url) as client,
        worker(tmp_path, backend) as base,
    ):
        job_id = client.post("/jobs", headers=ALICE, json={"prompt": "x"} | SMALL).json()["id"]
        for token in ("", "wrong-token-0123456789abcdefghijklmnop", ALICE_KEY):
            assert invoke(base, job_id, token)[0] == 401
        assert invoke(base, "not-a-uuid")[0] == 404
        assert invoke(base, "019a7c1e-4b2a-7c3d-9e8f-0123456789ab")[0] == 404
        assert fetch(f"{base}/health") == (200, b'{"status":"ok"}')
        assert client.get(f"/jobs/{job_id}", headers=ALICE).json()["status"] == "queued"
    assert backend.renders == 0


@pytest.mark.skipif(POSTGRES_URL is None, reason="CHRIEK_TEST_POSTGRES_URL not set")
def test_roles_starting_together_on_a_fresh_database_all_start(tmp_path):
    """API and worker processes start at once in a deployment; schema creation must not race."""
    url = make_url(base_env(tmp_path)["CHRIEK_DATABASE_URL"])
    barrier = threading.Barrier(6)
    errors: list[BaseException] = []

    def start() -> None:
        barrier.wait()
        try:
            JobStore(url).close()
        except BaseException as error:  # noqa: BLE001 - collected for the assertion
            errors.append(error)

    threads = [threading.Thread(target=start) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert errors == []
