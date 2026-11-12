"""Shared HTTP test helpers. Tests talk to chriek through its HTTP API (and CLI)."""

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
from fastapi.testclient import TestClient

from chriek.api import create_app
from chriek.backend import FakeBackend
from chriek.config import Settings

ALICE_KEY = "alice-key-0123456789abcdefghijklmnopqrstuvwxyz"
BOB_KEY = "bob-key-0123456789abcdefghijklmnopqrstuvwxyz"
SIGNING_SECRET = "signing-secret-0123456789abcdefghijklmnop"
ALICE = {"Authorization": f"Bearer {ALICE_KEY}"}
BOB = {"Authorization": f"Bearer {BOB_KEY}"}
TERMINAL = {"succeeded", "failed"}
SMALL = {"width": 16, "height": 16}  # cheapest valid image size


def base_env(tmp_path: Path) -> dict[str, str]:
    return {
        "CHRIEK_DB_PATH": str(tmp_path / "db" / "chriek.db"),
        "CHRIEK_STORAGE_DIR": str(tmp_path / "images"),
        "CHRIEK_API_KEYS": f"alice:{ALICE_KEY},bob:{BOB_KEY}",
        "CHRIEK_SIGNING_SECRET": SIGNING_SECRET,
    }


def proxy_env(tmp_path: Path) -> dict[str, str]:
    """Proxy mode as an operator configures it: no API keys at all."""
    env = base_env(tmp_path) | {"CHRIEK_AUTH_MODE": "proxy"}
    del env["CHRIEK_API_KEYS"]
    return env


class FakeClock:
    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class GatedBackend(FakeBackend):
    """Fake backend whose renders block until `release()`; records concurrency and order."""

    def __init__(self) -> None:
        self._gate = threading.Event()
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.prompts: list[str] = []
        self.rendering = threading.Event()

    def release(self) -> None:
        self._gate.set()

    def render(self, prompt, width, height, seed):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.prompts.append(prompt)
        self.rendering.set()
        try:
            assert self._gate.wait(10), "gated render never released"
            return super().render(prompt, width, height, seed)
        finally:
            with self._lock:
                self.active -= 1


class FailingBackend(FakeBackend):
    """Fake backend that fails only render number `fail_index` (0-based, counted across all
    jobs, never reset); counts renders. Jobs rendered after that one failure succeed."""

    def __init__(self, fail_index: int) -> None:
        self.fail_index = fail_index
        self.renders = 0

    def render(self, prompt, width, height, seed):
        self.renders += 1
        if self.renders - 1 == self.fail_index:
            raise RuntimeError("injected backend failure")
        return super().render(prompt, width, height, seed)


class CountingBackend(FakeBackend):
    def __init__(self) -> None:
        self.renders = 0

    def render(self, prompt, width, height, seed):
        self.renders += 1
        return super().render(prompt, width, height, seed)


class StepBackend(FakeBackend):
    """Fake backend whose renders each wait for one `step()`; `wait_started(n)` blocks until
    `n` renders have begun. Several apps may share one instance."""

    def __init__(self) -> None:
        self._steps = threading.Semaphore(0)
        self._started = threading.Condition()
        self.prompts: list[str] = []

    def step(self, renders: int = 1) -> None:
        self._steps.release(renders)

    def wait_started(self, renders: int) -> None:
        with self._started:
            assert self._started.wait_for(lambda: len(self.prompts) >= renders, 10), (
                f"only {len(self.prompts)} of {renders} renders started"
            )

    def render(self, prompt, width, height, seed):
        with self._started:
            self.prompts.append(prompt)
            self._started.notify_all()
        assert self._steps.acquire(timeout=10), "render step never released"
        return super().render(prompt, width, height, seed)


@contextmanager
def running(
    tmp_path: Path,
    *,
    backend=None,
    clock=None,
    env_base: Callable[[Path], dict[str, str]] = base_env,
    **env: str,
) -> Iterator[TestClient]:
    settings = Settings.from_env(env_base(tmp_path) | env)
    app = create_app(settings, backend=backend or FakeBackend(), clock=clock or time.time)
    with TestClient(app) as client:
        yield client


def wait_for(client: TestClient, job_id: str, statuses=TERMINAL, headers=ALICE) -> dict:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}", headers=headers).json()
        if body["status"] in statuses:
            return body
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never reached {statuses}")


def post_job(client: TestClient, prompt: str = "a lighthouse", headers=ALICE, **fields):
    return client.post("/jobs", headers=headers, json={"prompt": prompt} | fields)


def submit(client: TestClient, headers=ALICE, **fields) -> dict:
    response = post_job(client, headers=headers, **fields)
    assert response.status_code == 202, response.text
    return response.json()


def with_query(url: str, **changes: str) -> str:
    parts = urlsplit(url)
    query = {key: values[0] for key, values in parse_qs(parts.query).items()} | changes
    return parts._replace(query=urlencode(query)).geturl()


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with running(tmp_path) as test_client:
        yield test_client
