"""Shared HTTP test helpers. Tests talk to chriek only through its HTTP API (and CLI)."""

import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

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
# Text a backend exception could carry (paths, credentials, signed URLs); never public.
BACKEND_SECRET = (
    "injected backend failure at /srv/secret-weights/model.bin "
    "token=hf_SENTINEL0123 https://user:p4ss@example.invalid/x?sig=SENTINEL"
)


def base_env(tmp_path: Path) -> dict[str, str]:
    return {
        "CHRIEK_DB_PATH": str(tmp_path / "db" / "chriek.db"),
        "CHRIEK_STORAGE_DIR": str(tmp_path / "images"),
        "CHRIEK_API_KEYS": f"alice:{ALICE_KEY},bob:{BOB_KEY}",
        "CHRIEK_SIGNING_SECRET": SIGNING_SECRET,
        # TestClient's origin; `chriek serve` tests replace it with the served port.
        "CHRIEK_PUBLIC_URL": "http://testserver",
    }


def connect_db(tmp_path: Path) -> closing[sqlite3.Connection]:
    return closing(sqlite3.connect(base_env(tmp_path)["CHRIEK_DB_PATH"]))


def db_dump(tmp_path: Path) -> str:
    with connect_db(tmp_path) as db:
        return "\n".join(db.iterdump())


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
            raise RuntimeError(BACKEND_SECRET) from OSError(f"cause: {BACKEND_SECRET}")
        return super().render(prompt, width, height, seed)


class CountingBackend(FakeBackend):
    def __init__(self) -> None:
        self.renders = 0

    def render(self, prompt, width, height, seed):
        self.renders += 1
        return super().render(prompt, width, height, seed)


@contextmanager
def running(tmp_path: Path, *, backend=None, clock=None, **env: str) -> Iterator[TestClient]:
    settings = Settings.from_env(base_env(tmp_path) | env)
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


def submit(client: TestClient, headers=ALICE, **fields) -> dict:
    response = client.post("/jobs", headers=headers, json={"prompt": "a lighthouse"} | fields)
    assert response.status_code == 202, response.text
    return response.json()


def assert_nothing_queued(client: TestClient, backend: CountingBackend) -> None:
    """A valid job submitted afterwards is the first and only one the worker renders."""
    valid = submit(client, width=16, height=16)
    assert wait_for(client, valid["id"])["status"] == "succeeded"
    assert backend.renders == 1


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with running(tmp_path) as test_client:
        yield test_client
