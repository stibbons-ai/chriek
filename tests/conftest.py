"""Shared test helpers. Most tests talk to chriek through its HTTP API or the installed CLI;
others check configuration, the store, the worker or the database directly, including where
storage privacy must hold without the CLI (see README.md).

Every test runs on SQLite unless CHRIEK_TEST_POSTGRES_URL names a disposable Postgres server;
then each test not marked `sqlite` gets its own fresh database there, dropped afterwards.

Images go to a local directory unless CHRIEK_TEST_S3_URL
(`http://ACCESS_KEY:SECRET@host:port/bucket`) names an existing S3-compatible service; then each
test not marked `local_storage` stores below its own key prefix in that bucket (created if
missing), and only that prefix is emptied afterwards.
"""

import io
import os
import stat
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import boto3
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import URL, Connection, Engine, create_engine, make_url, text

from chriek.api import create_app
from chriek.backend import FakeBackend
from chriek.config import Settings

POSTGRES_URL = os.environ.get("CHRIEK_TEST_POSTGRES_URL")
S3_URL = os.environ.get("CHRIEK_TEST_S3_URL")
S3_BUCKET = urlsplit(S3_URL).path.strip("/") if S3_URL else None

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
SMALL = {"width": 16, "height": 16}  # cheapest valid image size


_postgres_database_url: str | None = None  # the running test's own Postgres database
_s3_prefix: str | None = None  # the running test's own key prefix in the S3 test bucket


def base_env(tmp_path: Path) -> dict[str, str]:
    env = {
        "CHRIEK_DB_PATH": str(tmp_path / "db" / "chriek.db"),
        "CHRIEK_STORAGE_DIR": str(tmp_path / "images"),
        "CHRIEK_API_KEYS": f"alice:{ALICE_KEY},bob:{BOB_KEY}",
        "CHRIEK_SIGNING_SECRET": SIGNING_SECRET,
        # TestClient's origin; `chriek serve` tests replace it with the served port.
        "CHRIEK_PUBLIC_URL": "http://testserver",
    }
    if _postgres_database_url:
        # CHRIEK_DB_PATH stays set: the URL must take precedence over it.
        env["CHRIEK_DATABASE_URL"] = _postgres_database_url
    if _s3_prefix:
        del env["CHRIEK_STORAGE_DIR"]
        env |= s3_env(_s3_prefix)
    return env


def s3_env(prefix: str) -> dict[str, str]:
    """CHRIEK_* settings for the S3 test service, storing below `prefix`."""
    parts = urlsplit(S3_URL)
    return {
        "CHRIEK_STORAGE": "s3",
        "CHRIEK_S3_ENDPOINT_URL": f"{parts.scheme}://{parts.hostname}:{parts.port}",
        "CHRIEK_S3_BUCKET": S3_BUCKET,
        "CHRIEK_S3_PREFIX": prefix,
        "CHRIEK_S3_ACCESS_KEY_ID": parts.username,
        "CHRIEK_S3_SECRET_ACCESS_KEY": parts.password,
    }


def pytest_report_header() -> list[str]:
    return [
        f"chriek test database: {'SQLite' if POSTGRES_URL is None else 'Postgres'}",
        f"chriek test image storage: {'local' if S3_URL is None else 'S3'}",
    ]


@cache
def _s3():
    env = s3_env("")
    client = boto3.client(
        "s3",
        endpoint_url=env["CHRIEK_S3_ENDPOINT_URL"],
        region_name="us-east-1",
        aws_access_key_id=env["CHRIEK_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=env["CHRIEK_S3_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    ensure_bucket(client, S3_BUCKET)
    return client


def ensure_bucket(client, bucket: str) -> None:
    """Create `bucket` only when it is missing; any other error (403, outage) propagates."""
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as error:
        if error.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
            raise
        client.create_bucket(Bucket=bucket)


@pytest.fixture
def s3_client():
    return _s3()


def s3_keys(client, prefix: str) -> list[str]:
    """Every key below `prefix/`, relative to it, sorted."""
    pages = client.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=f"{prefix}/")
    items = (item for page in pages for item in page.get("Contents", []))
    return sorted(item["Key"].removeprefix(f"{prefix}/") for item in items)


def stored_images(tmp_path: Path) -> list[str]:
    """What this test stored in image storage, relative to its root: local files and
    directories below `tmp_path/images`, or the objects below this test's S3 prefix."""
    if _s3_prefix:
        return s3_keys(_s3(), _s3_prefix)
    root = tmp_path / "images"
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def read_stored(tmp_path: Path, name: str) -> bytes:
    """One stored image by its name relative to the storage root, such as `<id>/0.png`."""
    if _s3_prefix:
        return _s3().get_object(Bucket=S3_BUCKET, Key=f"{_s3_prefix}/{name}")["Body"].read()
    return (tmp_path / "images" / name).read_bytes()


def fetch(url: str) -> tuple[int, bytes]:
    """A real HTTP GET: status and body, also for error statuses."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 - test URLs
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def download(client: TestClient, url: str) -> tuple[int, bytes]:
    """GET a download URL: chriek's own route through the test client, an S3 link directly."""
    if url.startswith(f"{client.base_url}/"):
        response = client.get(url)
        return response.status_code, response.content
    return fetch(url)


@pytest.fixture(autouse=True)
def storage(request: pytest.FixtureRequest) -> Iterator[str | None]:
    """This test's own S3 key prefix, or None with local image storage."""
    global _s3_prefix
    if S3_URL is None or request.node.get_closest_marker("local_storage"):
        yield None
        return
    prefix = f"chriek-test/{uuid.uuid4().hex}"  # only this prefix's objects get deleted
    _s3()  # creates the test bucket before the first app checks it
    _s3_prefix = prefix
    yield prefix
    _s3_prefix = None
    for key in s3_keys(_s3(), prefix):
        _s3().delete_object(Bucket=S3_BUCKET, Key=f"{prefix}/{key}")


@pytest.fixture(scope="session")
def postgres_admin() -> Iterator[Engine]:
    engine = create_engine(make_url(POSTGRES_URL), isolation_level="AUTOCOMMIT")
    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def database(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[str | None]:
    """The name of this test's own Postgres database, or None on SQLite."""
    global _postgres_database_url
    if POSTGRES_URL is None or request.node.get_closest_marker("sqlite"):
        yield None
        return

    admin: Engine = request.getfixturevalue("postgres_admin")
    name = f"chriek_test_{uuid.uuid4().hex}"  # only databases this fixture created get dropped
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    _postgres_database_url = admin.url.set(database=name).render_as_string(hide_password=False)
    yield name
    _postgres_database_url = None
    with admin.connect() as connection:
        connection.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
    assert not (tmp_path / "db").exists(), "a Postgres test wrote a SQLite database"


@contextmanager
def connect_db(tmp_path: Path) -> Iterator[Connection]:
    """A transaction on this test's database, SQLite file or Postgres; commits on success."""
    env = base_env(tmp_path)
    url = env.get("CHRIEK_DATABASE_URL") or URL.create("sqlite", database=env["CHRIEK_DB_PATH"])
    engine = create_engine(url)
    try:
        with engine.begin() as db:
            yield db
    finally:
        engine.dispose()


def db_dump(tmp_path: Path) -> str:
    """Every stored row, in a stable order."""
    with connect_db(tmp_path) as db:
        return "\n".join(
            repr(tuple(row))
            for query in (
                "SELECT * FROM jobs ORDER BY seq",
                "SELECT * FROM images ORDER BY 1, 2",
                "SELECT * FROM job_references ORDER BY 1, 2",
            )
            for row in db.execute(text(query))
        )


def modes(root: Path) -> dict[str, int]:
    """Permission bits of every path below `root`, by relative path; links are not followed."""
    return {str(p.relative_to(root)): stat.S_IMODE(p.lstat().st_mode) for p in root.rglob("*")}


def tree_state(root: Path) -> dict[str, tuple]:
    """Mode, owner and content (or link target) of `root` and everything below it."""
    state = {}
    for path in [root, *root.rglob("*")]:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            content = os.readlink(path)
        else:
            content = path.read_bytes() if stat.S_ISREG(info.st_mode) else None
        state[str(path.relative_to(root))] = (info.st_mode, info.st_uid, content)
    return state


def proxy_env(tmp_path: Path) -> dict[str, str]:
    """Proxy mode as an operator configures it: no API keys at all. TestClient sends
    `Host: testserver`, so it is allowed here explicitly; it is no production default."""
    env = base_env(tmp_path) | {"CHRIEK_AUTH_MODE": "proxy", "CHRIEK_ALLOWED_HOSTS": "testserver"}
    del env["CHRIEK_API_KEYS"]
    return env


def stored_owners(tmp_path: Path) -> list[str]:
    """Owner of every stored job, oldest first, read from this test's database."""
    with connect_db(tmp_path) as db:
        return list(db.scalars(text("SELECT owner FROM jobs ORDER BY seq")))


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

    def render(self, prompt, width, height, seed, references=()):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.prompts.append(prompt)
        self.rendering.set()
        try:
            assert self._gate.wait(10), "gated render never released"
            return super().render(prompt, width, height, seed, references)
        finally:
            with self._lock:
                self.active -= 1


class PausingBackend(FakeBackend):
    """Fake backend that waits for `resume` before rendering seed 1: in a seed-0 job, the
    second image, once the first is on disk."""

    def __init__(self) -> None:
        self.paused = threading.Event()
        self.resume = threading.Event()

    def render(self, prompt, width, height, seed, references=()):
        if seed == 1:
            self.paused.set()
            assert self.resume.wait(10), "paused render never resumed"
        return super().render(prompt, width, height, seed, references)


class FailingBackend(FakeBackend):
    """Fake backend that fails only render number `fail_index` (0-based, counted across all
    jobs, never reset); counts renders. Jobs rendered after that one failure succeed."""

    def __init__(self, fail_index: int) -> None:
        self.fail_index = fail_index
        self.renders = 0

    def render(self, prompt, width, height, seed, references=()):
        self.renders += 1
        if self.renders - 1 == self.fail_index:
            raise RuntimeError(BACKEND_SECRET) from OSError(f"cause: {BACKEND_SECRET}")
        return super().render(prompt, width, height, seed, references)


class CountingBackend(FakeBackend):
    def __init__(self) -> None:
        self.renders = 0

    def render(self, prompt, width, height, seed, references=()):
        self.renders += 1
        return super().render(prompt, width, height, seed, references)


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

    def render(self, prompt, width, height, seed, references=()):
        with self._started:
            self.prompts.append(prompt)
            self._started.notify_all()
        assert self._steps.acquire(timeout=10), "render step never released"
        return super().render(prompt, width, height, seed, references)


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


def assert_nothing_queued(client: TestClient, backend: CountingBackend) -> None:
    """A valid job submitted afterwards is the first and only one the worker renders."""
    valid = submit(client, width=16, height=16)
    assert wait_for(client, valid["id"])["status"] == "succeeded"
    assert backend.renders == 1


def with_query(url: str, **changes: str) -> str:
    parts = urlsplit(url)
    query = {key: values[0] for key, values in parse_qs(parts.query).items()} | changes
    return parts._replace(query=urlencode(query)).geturl()


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with running(tmp_path) as test_client:
        yield test_client


def make_image_bytes(
    color: tuple[int, int, int] = (20, 40, 60),
    size: tuple[int, int] = (16, 16),
    fmt: str = "PNG",
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()
