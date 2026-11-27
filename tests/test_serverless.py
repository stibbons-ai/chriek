"""Serverless roles: `chriek api` dispatches each job ID over HTTP to `chriek worker`.

Configuration cases run everywhere. The walkthroughs need the shared Postgres database and S3
bucket the roles require, so they run only with both CHRIEK_TEST_POSTGRES_URL and
CHRIEK_TEST_S3_URL (see conftest.py); the worker serves real HTTP on a loopback port.
"""

import io
import json
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import anyio
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
from sqlalchemy.exc import OperationalError
from test_references import _fault_in_create, post_multipart, submit_multipart

from chriek import api as api_module
from chriek import cli as cli_module
from chriek import dispatch as dispatch_module
from chriek.api import create_app, create_worker_app
from chriek.cli import main
from chriek.config import (
    SETTING_NAMES,
    WORKER_SETTING_NAMES,
    ConfigError,
    Dispatch,
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
needs_postgres = pytest.mark.skipif(
    POSTGRES_URL is None, reason="concurrent claims need CHRIEK_TEST_POSTGRES_URL"
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
        ({"CHRIEK_DISPATCH_TOKEN": WORKER_TOKEN}, None, "CHRIEK_WORKER_TOKEN must differ"),
        *(
            (
                {"CHRIEK_DISPATCH_HEADERS": json.dumps({"X-Api-Key": value})},
                None,
                "CHRIEK_WORKER_TOKEN must differ",
            )
            for value in (WORKER_TOKEN, f"Key {WORKER_TOKEN}")
        ),
        ({"CHRIEK_BACKEND": "fake"}, None, "CHRIEK_BACKEND does not apply to chriek api"),
        ({"CHRIEK_OFFLOAD": "none"}, None, "CHRIEK_OFFLOAD does not apply to chriek api"),
        ({"CHRIEK_DISPATCH_URL": "ftp://worker.example"}, None, "CHRIEK_DISPATCH_URL must be"),
        *(
            ({"CHRIEK_DISPATCH_HEADERS": raw}, None, "must be a JSON object")
            for raw in ("Prefer: respond-async", '{"Prefer": 1}', "[]", '"x"')
        ),
        *(
            (
                {"CHRIEK_DISPATCH_HEADERS": json.dumps({name: PROVIDER_TOKEN})},
                None,
                "must not set Authorization",
            )
            for name in (
                "authorization",
                "CONTENT-TYPE",
                "idempotency-KEY",
                "x-Chriek-Worker-Token",
            )
        ),
        (
            {"CHRIEK_DISPATCH_HEADERS": json.dumps({"X-Key": f"{PROVIDER_TOKEN}\r\nX: y"})},
            None,
            "values printable ASCII",
        ),
        ({"CHRIEK_DISPATCH_TOKEN": f"{PROVIDER_TOKEN} x"}, None, "CHRIEK_DISPATCH_TOKEN must"),
        *(
            (
                {"CHRIEK_DISPATCH_TIMEOUT_SECONDS": raw},
                None,
                "CHRIEK_DISPATCH_TIMEOUT_SECONDS must be a number of seconds above 0 and at most "
                "604800 (7 days)",
            )
            for raw in ("0", "-1", "nan", "inf", "30s", "", "604800.5", "1e300")
        ),
        *(
            ({"CHRIEK_DISPATCH_ID_HEADER": name}, None, "must be one HTTP header token")
            for name in ("", f"{PROVIDER_TOKEN} x", "X-Id\r\nHost: evil.example", "X-Id:")
        ),
        *(
            ({"CHRIEK_DISPATCH_ID_HEADER": name}, None, "must not name a header chriek sets")
            for name in (
                "idempotency-key",
                "X-CHRIEK-WORKER-TOKEN",
                "authorization",
                "content-type",
                "host",
                "Content-Length",
                "transfer-encoding",
                "Connection",
                "prefer",  # configured in CHRIEK_DISPATCH_HEADERS below
            )
        ),
        (
            {
                "CHRIEK_DISPATCH_HEADERS": json.dumps({PROVIDER_TOKEN: "x", "Prefer": "x"}),
                "CHRIEK_DISPATCH_ID_HEADER": PROVIDER_TOKEN.upper(),
            },
            None,
            "one in CHRIEK_DISPATCH_HEADERS",
        ),
    ],
)
def test_api_role_rejects_configuration_without_leaking_secrets(
    tmp_path, changes, removed, message
):
    prefer = {"CHRIEK_DISPATCH_HEADERS": json.dumps({"Prefer": "respond-async"})}
    env = config_api_env(tmp_path, **(prefer | changes))
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
            CHRIEK_DISPATCH_URL="https://worker.example/hooks/path-token-SENTINEL/invoke",
            CHRIEK_DISPATCH_TOKEN=PROVIDER_TOKEN,
            CHRIEK_DISPATCH_HEADERS=json.dumps({"Prefer": "respond-async"}),
            CHRIEK_DISPATCH_TIMEOUT_SECONDS="2.5",
            CHRIEK_DISPATCH_ID_HEADER="X-Inference-Id",
        ),
        Role.API,
    )
    assert settings.dispatch.headers == {
        "Prefer": "respond-async",
        "Authorization": f"Bearer {PROVIDER_TOKEN}",
    }
    assert settings.dispatch.timeout_seconds == 2.5
    assert settings.dispatch.id_header == "X-Inference-Id"
    for secret in ("path-token-SENTINEL", PROVIDER_TOKEN, WORKER_TOKEN):
        assert secret not in repr(settings)
    default = Settings.from_env(config_api_env(tmp_path), Role.API).dispatch
    assert (default.timeout_seconds, default.id_header) == (30, None)
    maximum = config_api_env(tmp_path, CHRIEK_DISPATCH_TIMEOUT_SECONDS="604800")
    assert Settings.from_env(maximum, Role.API).dispatch.timeout_seconds == 604800


class RecordingStore:
    """Answers `fail_queued` as if the job were still queued (`queued`) or already claimed."""

    def __init__(self, queued: bool) -> None:
        self.queued = queued
        self.errors: list[str] = []

    def fail_queued(self, _job_id: str, error: str, _now: float) -> bool:
        self.errors.append(error)
        return self.queued


def dispatch_raising(monkeypatch, error: Exception, store: RecordingStore) -> None:
    class Opener:
        def open(self, *_args, **_kwargs):
            raise error

    monkeypatch.setattr(dispatch_module.urllib.request, "build_opener", lambda *_: Opener())
    settings = Dispatch("https://worker.example/invoke", WORKER_TOKEN, {}, timeout_seconds=2.5)
    dispatch_module.dispatch(settings, store, "job-1", lambda: 0.0)


@pytest.mark.parametrize(
    "error",
    [TimeoutError("timed out"), urllib.error.URLError(TimeoutError("timed out"))],
    ids=["read", "connect-wrapped-by-urllib"],
)
def test_dispatch_timeout_is_named_as_such(monkeypatch, error):
    store = RecordingStore(queued=True)
    dispatch_raising(monkeypatch, error, store)
    assert store.errors == [
        "dispatch failed: the worker endpoint gave no answer within 2.5 seconds"
    ]


@pytest.mark.parametrize("queued", [True, False])
def test_dispatch_logs_an_error_only_for_a_recorded_failure(monkeypatch, caplog, queued):
    dispatch_raising(monkeypatch, TimeoutError("timed out"), RecordingStore(queued))
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == (1 if queued else 0)


class BrokenStore(RecordingStore):
    def fail_queued(self, job_id: str, error: str, now: float) -> bool:
        super().fail_queued(job_id, error, now)
        raise OperationalError("UPDATE", {}, ConnectionError("connection lost"))


def test_dispatch_logs_but_contains_a_failure_to_record_the_delivery_failure(monkeypatch, caplog):
    store = BrokenStore(queued=True)
    dispatch_raising(monkeypatch, TimeoutError("timed out"), store)  # must not raise
    assert len(store.errors) == 1
    (record,) = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert record.exc_info is not None
    assert "recording that failed too" in record.getMessage()


@pytest.mark.parametrize(
    "name",
    [
        "CHRIEK_DISPATCH_URL",
        "CHRIEK_DISPATCH_TOKEN",
        "CHRIEK_DISPATCH_HEADERS",
        "CHRIEK_DISPATCH_ID_HEADER",
        "CHRIEK_DISPATCH_TIMEOUT_SECONDS",
        "CHRIEK_WORKER_TOKEN",
    ],
)
def test_home_role_refuses_serverless_settings(tmp_path, name):
    with pytest.raises(ConfigError, match=f"{name} does not apply to chriek serve"):
        Settings.from_env(base_env(tmp_path) | {name: WORKER_TOKEN})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
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


@pytest.mark.parametrize("name", sorted(SETTING_NAMES - WORKER_SETTING_NAMES))
def test_worker_role_refuses_every_api_and_dispatch_setting(name):
    """Client API keys, the signing secret and the provider credential never reach the GPU."""
    env = SHARED | {"CHRIEK_WORKER_TOKEN": WORKER_TOKEN, name: PROVIDER_TOKEN}
    with pytest.raises(ConfigError, match=f"{name} does not apply to chriek worker") as error:
        WorkerSettings.from_env(env)
    assert PROVIDER_TOKEN not in str(error.value)


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
    for name in ("create_app", "create_worker_app"):  # each opens the database and the bucket
        monkeypatch.setattr(cli_module, name, lambda *a, **kw: pytest.fail("storage opened"))

    assert main([command]) == 2
    error = capsys.readouterr().err
    assert error.startswith(
        f"chriek: configuration error: chriek {command} requires CHRIEK_DATABASE_URL"
    )
    assert WORKER_TOKEN not in error
    assert not (tmp_path / "db").exists()  # the SQLite file `chriek api` was given, unopened


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
    worker (recording its answer), and answers with `status` and, if given, `location`. With
    a `gate`, each delivery waits for the gate to open first."""

    def __init__(
        self,
        status: int = 202,
        forward_to: str | None = None,
        delay: float = 0.0,
        location: str | None = None,
        gate: threading.Event | None = None,
    ) -> None:
        self.deliveries: list[tuple[dict[str, str], dict]] = []
        self.arrived = threading.Semaphore(0)  # released once per delivery, before any gate
        self.answered = threading.Event()  # set once a delivery got its answer
        self.worker_answers: list[tuple[int, dict]] = []
        self.app = FastAPI()

        @self.app.post("/{path:path}")
        def deliver(request_headers: Request, body: dict) -> Response:
            headers = dict(request_headers.headers)
            self.deliveries.append((headers, body))
            self.arrived.release()
            if gate is not None:
                assert gate.wait(10), "gated delivery never released"
            time.sleep(delay)  # a cold start before the worker sees the job
            if forward_to:
                self.worker_answers.append(invoke(forward_to, body["job_id"]))
            self.answered.set()
            return Response(
                status_code=status, headers={"Location": location} if location else None
            )


def invoke(base: str, job_id: str, token: str = WORKER_TOKEN) -> tuple[int, dict]:
    return post_invoke(base, {"job_id": job_id}, token)


def post_invoke(base: str, payload: dict, token: str = WORKER_TOKEN) -> tuple[int, dict]:
    request = urllib.request.Request(  # noqa: S310 - loopback test server
        f"{base}/invoke",
        data=json.dumps(payload).encode(),
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
    assert "x-inference-id" not in headers  # only with CHRIEK_DISPATCH_ID_HEADER


@needs_services
def test_configured_id_header_carries_the_submitted_job_id(tmp_path):
    provider = Provider(status=202)
    with (
        serving(provider.app) as provider_url,
        api(tmp_path, provider_url, CHRIEK_DISPATCH_ID_HEADER="X-Inference-Id") as client,
    ):
        job_id = submit(client, **SMALL)["id"]
        assert provider.arrived.acquire(timeout=10), "dispatch never arrived"
    ((headers, body),) = provider.deliveries
    assert body == {"job_id": job_id}
    assert headers["x-inference-id"] == job_id
    assert headers["idempotency-key"] == job_id
    assert headers["prefer"] == "respond-async"
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
        # Nor does a worker cold-starting during the render.
        with worker(tmp_path, CountingBackend()):
            assert client.get(f"/jobs/{job_id}", headers=ALICE).json()["status"] == "running"
        backend.release()
        thread.join(20)
        assert wait_for(client, job_id)["status"] == "succeeded"

    assert results == [(200, {"id": job_id, "status": "succeeded"})]
    assert backend.prompts == ["a lighthouse"]


def queued_jobs(store: JobStore, count: int) -> list[str]:
    return [
        store.create(
            owner="alice",
            prompt="a lighthouse",
            image_count=1,
            seed=0,
            now=1.0,
            max_queued=count,
            **SMALL,
        ).id
        for _ in range(count)
    ]


@needs_postgres
def test_simultaneous_claims_of_one_job_on_separate_connections_claim_it_once(tmp_path):
    """Duplicate deliveries reaching warm workers at once: each claimant is its own store and
    connection, all released together by a barrier, per job."""
    url = make_url(base_env(tmp_path)["CHRIEK_DATABASE_URL"])
    claimants = 4
    stores = [JobStore(url) for _ in range(claimants)]
    start = threading.Barrier(claimants, timeout=10)  # a stuck claimant breaks it, not the test

    def claim(store: JobStore, job_id: str) -> bool:
        start.wait()
        return store.claim(job_id, 2.0) is not None

    try:
        job_ids = queued_jobs(stores[0], 50)
        for store in stores:  # connect before the first race, not during it
            store.status(job_ids[0])
        with ThreadPoolExecutor(claimants) as pool:
            for job_id in job_ids:
                wins = list(pool.map(claim, stores, [job_id] * claimants))
                assert wins.count(True) == 1, job_id
                assert stores[0].status(job_id) == "running"
    finally:
        start.abort()
        for store in stores:
            store.close()


@needs_postgres
def test_fail_queued_fails_only_a_still_queued_job(tmp_path):
    store = JobStore(make_url(base_env(tmp_path)["CHRIEK_DATABASE_URL"]))
    try:
        queued, claimed = queued_jobs(store, 2)
        assert store.claim(claimed, 2.0) is not None
        assert store.fail_queued(queued, "dispatch failed: x", 3.0) is True
        assert store.fail_queued(queued, "dispatch failed: y", 4.0) is False  # already failed
        assert store.fail_queued(claimed, "dispatch failed: x", 3.0) is False
        assert (store.status(queued), store.status(claimed)) == ("failed", "running")
    finally:
        store.close()


@needs_services
def test_render_failure_is_recorded_and_returned(tmp_path):
    with worker(tmp_path, FailingBackend(0)) as base:
        provider = Provider(status=200, forward_to=base)
        with serving(provider.app) as provider_url, api(tmp_path, provider_url) as client:
            job_id = submit(client, **SMALL)["id"]
            job = wait_for(client, job_id)
    assert provider.worker_answers == [(200, {"id": job_id, "status": "failed"})]
    assert job["status"] == "failed"
    assert job["error"] == "image 1 of 1 failed"
    assert job["images"] == []


@needs_services
@pytest.mark.parametrize("answer", [503, None])
def test_delivery_failure_fails_the_queued_job_without_secrets(tmp_path, caplog, answer):
    caplog.set_level(logging.INFO)
    secret_path = "/hooks/path-token-SENTINEL/invoke"
    if answer is None:
        # Port 1 lies below the ephemeral range, so no test server bound to port 0 can take it
        # meanwhile. A bound, non-listening socket would not do: macOS drops its SYNs (timeout).
        job = run_job(tmp_path, f"http://127.0.0.1:1{secret_path}")
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
    assert job["finished_at"] is not None
    assert (job["started_at"], job["run"]) == (None, None)  # no worker ever ran it
    assert f"dispatching job {job['id']} failed" in caplog.text
    for secret in ("SENTINEL", PROVIDER_TOKEN, WORKER_TOKEN):
        assert secret not in json.dumps(job)
        assert secret not in caplog.text


@needs_services
@pytest.mark.parametrize("answer", [302, 303])
def test_dispatch_never_follows_a_redirect(tmp_path, answer):
    """Following it would resend the worker token and provider credential to the target."""
    hits: list[str] = []  # only methods: nothing credential-bearing is recorded
    destination = FastAPI()

    @destination.api_route("/{path:path}", methods=["GET", "HEAD", "POST", "PUT"])
    def count(request: Request) -> Response:
        hits.append(request.method)
        return Response(status_code=200)

    with serving(destination) as destination_url:
        provider = Provider(status=answer, location=f"{destination_url}/elsewhere")
        with serving(provider.app) as provider_url:
            job = run_job(tmp_path, provider_url)

    assert len(provider.deliveries) == 1
    assert hits == []
    assert (job["status"], job["error"]) == (
        "failed",
        f"dispatch failed: the worker endpoint answered HTTP {answer}",
    )


@needs_services
def test_submission_answers_202_while_the_provider_still_holds_the_delivery(tmp_path):
    gate = threading.Event()
    provider = Provider(status=202, gate=gate)
    try:
        with serving(provider.app) as provider_url:
            app = create_app(Settings.from_env(api_env(tmp_path, provider_url), Role.API))
            with serving(app) as base:
                request = urllib.request.Request(  # noqa: S310 - loopback test server
                    f"{base}/jobs",
                    data=json.dumps({"prompt": "x"} | SMALL).encode(),
                    headers={"Content-Type": "application/json"} | ALICE,
                    method="POST",
                )
                # A dispatch before the response would hold this request until the gate opens.
                with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
                    status, job = response.status, json.loads(response.read())
                assert (status, job["status"]) == (202, "queued")
                assert not gate.is_set()

                gate.set()
                assert provider.answered.wait(10)
        assert [body for _headers, body in provider.deliveries] == [{"job_id": job["id"]}]
    finally:
        gate.set()


class RecordingLock:
    """Stands in for the worker's render lock and counts attempts to take it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._changed = threading.Condition()
        self.attempts = 0

    def __enter__(self) -> None:
        with self._changed:
            self.attempts += 1
            self._changed.notify_all()
        self._lock.acquire()

    def __exit__(self, *_exc) -> None:
        self._lock.release()

    def wait_for_attempts(self, count: int) -> bool:
        with self._changed:
            return self._changed.wait_for(lambda: self.attempts >= count, 10)


@needs_services
def test_one_worker_process_renders_one_job_at_a_time(tmp_path, monkeypatch):
    lock = RecordingLock()
    monkeypatch.setattr(api_module, "threading", SimpleNamespace(Lock=lambda: lock))
    backend = GatedBackend()
    results: dict[str, tuple[int, dict]] = {}
    with (
        serving(Provider(status=202).app) as provider_url,
        api(tmp_path, provider_url) as client,
        worker(tmp_path, backend) as base,
    ):
        a = submit(client, prompt="job a", **SMALL)["id"]
        b = submit(client, prompt="job b", **SMALL)["id"]
        calls = [("a", a), ("duplicate of a", a), ("b", b)]
        threads = [
            threading.Thread(target=lambda n=name, j=job_id: results.update({n: invoke(base, j)}))
            for name, job_id in calls
        ]
        threads[0].start()
        assert backend.rendering.wait(10)
        for thread in threads[1:]:
            thread.start()
        assert lock.wait_for_attempts(3)  # both later invocations reached the lock
        backend.release()
        for thread in threads:
            thread.join(20)

    assert results == {
        "a": (200, {"id": a, "status": "succeeded"}),
        "duplicate of a": (200, {"id": a, "status": "succeeded"}),
        "b": (200, {"id": b, "status": "succeeded"}),
    }
    assert backend.max_active == 1
    assert backend.prompts == ["job a", "job b"]


def with_thread_tokens(app, tokens: int):
    """Wrap `app` so its event loop's default threadpool has `tokens` tokens (AnyIO: 40)."""

    async def wrapped(scope, receive, send):
        anyio.to_thread.current_default_thread_limiter().total_tokens = tokens
        await app(scope, receive, send)

    return wrapped


@needs_services
def test_worker_health_answers_while_invocations_hold_every_thread(tmp_path, monkeypatch):
    lock = RecordingLock()
    monkeypatch.setattr(api_module, "threading", SimpleNamespace(Lock=lambda: lock))
    backend = GatedBackend()
    threads: list[threading.Thread] = []
    app = create_worker_app(WorkerSettings.from_env(worker_env(tmp_path)), backend=backend)
    with (
        serving(Provider(status=202).app) as provider_url,
        api(tmp_path, provider_url) as client,
        serving(with_thread_tokens(app, 2)) as base,
    ):
        try:
            job_id = submit(client, **SMALL)["id"]
            # One invocation renders, a duplicate waits for the lock: both tokens are held.
            threads = [threading.Thread(target=invoke, args=(base, job_id)) for _ in range(2)]
            for thread in threads:
                thread.start()
            assert backend.rendering.wait(10)
            assert lock.wait_for_attempts(2)
            with urllib.request.urlopen(f"{base}/health", timeout=5) as response:  # noqa: S310
                assert (response.status, response.read()) == (200, b'{"status":"ok"}')
        finally:
            backend.release()
            for thread in threads:
                thread.join(20)


@needs_services
def test_api_health_answers_while_dispatches_hold_every_thread(tmp_path):
    gate = threading.Event()
    provider = Provider(status=202, gate=gate)
    with serving(provider.app) as provider_url:
        app = create_app(Settings.from_env(api_env(tmp_path, provider_url), Role.API))
        with serving(with_thread_tokens(app, 2)) as base:
            try:
                for _ in range(2):  # each accepted job's dispatch then holds one of the 2 tokens
                    request = urllib.request.Request(  # noqa: S310 - loopback test server
                        f"{base}/jobs",
                        data=json.dumps({"prompt": "a lighthouse"} | SMALL).encode(),
                        headers=ALICE | {"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
                        assert response.status == 202
                for _ in range(2):
                    assert provider.arrived.acquire(timeout=10), "dispatch never arrived"
                with urllib.request.urlopen(f"{base}/health", timeout=3) as response:  # noqa: S310
                    assert (response.status, response.read()) == (200, b'{"status":"ok"}')
                assert not provider.answered.is_set()  # both dispatches still held their tokens
            finally:
                gate.set()


def _connection_lost(*_args) -> None:
    raise OperationalError("SELECT", {}, ConnectionError("connection lost"))


@needs_services
def test_worker_answers_500_when_the_result_cannot_be_saved(tmp_path, monkeypatch):
    backend = CountingBackend()
    with (
        serving(Provider(status=202).app) as provider_url,
        api(tmp_path, provider_url) as client,
        worker(tmp_path, backend) as base,
    ):
        job_id = submit(client, **SMALL)["id"]
        monkeypatch.setattr(JobStore, "succeed", _connection_lost)
        assert invoke(base, job_id) == (500, {"detail": "job result could not be saved"})
        assert client.get(f"/jobs/{job_id}", headers=ALICE).json()["status"] == "running"
    assert backend.renders == 1


@needs_services
@pytest.mark.parametrize(
    ("event_name", "verify_fails", "delivered"),
    [("after_commit", False, True), ("before_commit", False, False), ("after_commit", True, False)],
    ids=["committed", "never-committed", "committed-but-unverifiable"],
)
def test_unknown_commit_outcome_delivers_only_a_job_found_committed(
    tmp_path, monkeypatch, event_name, verify_fails, delivered
):
    backend = ReferenceBackend()
    if verify_fails:  # the fresh read after the failed COMMIT fails too
        monkeypatch.setattr(JobStore, "status", _connection_lost)
    with worker(tmp_path, backend) as base:
        provider = Provider(status=200, forward_to=base)
        with serving(provider.app) as provider_url:
            app = create_app(Settings.from_env(api_env(tmp_path, provider_url), Role.API))
            with (
                _fault_in_create(monkeypatch, event_name) as fired,
                TestClient(app, raise_server_exceptions=False) as client,
            ):
                response = post_multipart(
                    client,
                    data={"prompt": "lost ack", "width": "16", "height": "16"},
                    files=[("references", ("r.png", make_image_bytes((9, 9, 9)), "image/png"))],
                )

    # The submission's own error stands, whatever the fresh read found.
    assert fired == [event_name]
    assert response.status_code == 500
    assert "location" not in response.headers
    if not delivered:
        assert provider.deliveries == []
        assert backend.renders == 0
        return
    ((_headers, body),) = provider.deliveries
    job_id = body["job_id"]
    assert provider.worker_answers == [(200, {"id": job_id, "status": "succeeded"})]
    assert backend.references == [(16 * 16 * 3,)]  # the kept reference, read by the worker
    with api(tmp_path, provider_url) as client:
        job = wait_for(client, job_id)
    assert (job["status"], job["params"]["reference_count"]) == ("succeeded", 1)


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
@pytest.mark.parametrize(
    ("timeout", "status", "error", "renders"),
    [
        (
            "0.3",
            "failed",
            "dispatch failed: the worker endpoint gave no answer within 0.3 seconds",
            0,
        ),
        ("15", "succeeded", None, 1),
    ],
    ids=["deadline-before-claim", "deadline-covers-cold-start-and-render"],
)
def test_synchronous_endpoint_with_a_slow_cold_start(tmp_path, timeout, status, error, renders):
    """A synchronous provider holds the request through a 1 s cold start, then the render."""
    backend = CountingBackend()
    with worker(tmp_path, backend) as base:
        provider = Provider(status=200, forward_to=base, delay=1.0)
        with (
            serving(provider.app) as provider_url,
            api(tmp_path, provider_url, CHRIEK_DISPATCH_TIMEOUT_SECONDS=timeout) as client,
        ):
            job_id = submit(client, **SMALL)["id"]
            job = wait_for(client, job_id)
            assert (job["status"], job["error"]) == (status, error)
        # Serving stops only once the delivery reached the worker, also the one that outlived
        # its deadline: the worker found that job failed and rendered nothing.
        assert provider.worker_answers == [(200, {"id": job_id, "status": status})]
        with api(tmp_path, provider_url) as client:
            job = wait_for(client, job_id)
    assert (job["status"], job["error"], len(job["images"])) == (status, error, renders)
    assert backend.renders == renders


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

        # Authenticated but invalid: a useful 422 that never echoes the submitted input.
        status, body = post_invoke(base, {"job_id": job_id, "extra": "INPUT-SENTINEL"})
        assert status == 422
        assert body["detail"][0]["loc"] == ["body", "extra"]
        assert "INPUT-SENTINEL" not in json.dumps(body)

        # The ID is matched in its canonical form: an upper-case delivery runs the job.
        assert invoke(base, job_id.upper()) == (200, {"id": job_id, "status": "succeeded"})
        assert client.get(f"/jobs/{job_id}", headers=ALICE).json()["status"] == "succeeded"
    assert backend.renders == 1


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
