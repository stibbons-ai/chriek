"""Storage privacy at the store, worker and app boundaries, without the CLI."""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import modes, running, submit, tree_state, wait_for

from chriek.backend import FakeBackend
from chriek.storage import StorageError
from chriek.store import JobStore
from chriek.worker import Worker

# Runs in a child process so its umask never leaks into other tests.
DIRECT_ENTRY_POINTS = """
import json, sys, threading, time
from pathlib import Path

from conftest import modes
from chriek.backend import FakeBackend
from chriek.store import SUCCEEDED, JobStore
from chriek.worker import Worker

root = Path(sys.argv[1])
db_path = root / "data" / "db" / "chriek.db"
paused, resume = threading.Event(), threading.Event()


class PausingBackend(FakeBackend):
    def render(self, prompt, width, height, seed):
        if seed == 1:  # the second image: the first one is already on disk
            paused.set()
            resume.wait(10)
        return super().render(prompt, width, height, seed)


store = JobStore(db_path)
job = store.create(owner="alice", prompt="p", image_count=2, width=16, height=16, seed=0, now=0.0)
worker = Worker(store, PausingBackend(), root / "data" / "images", time.time)
worker.start()
assert paused.wait(10)
seen = {"job": job.id, "paused": modes(root)}
resume.set()
deadline = time.monotonic() + 10
while store.get("alice", job.id).status != SUCCEEDED:
    assert time.monotonic() < deadline
    time.sleep(0.02)
worker.stop(10)
seen["done"] = modes(root)
store.close()  # the last connection removes the WAL and SHM sidecars
seen["closed"] = modes(root)
store = JobStore(db_path)
store.create(owner="alice", prompt="q", image_count=1, width=16, height=16, seed=0, now=1.0)
seen["reopened"] = modes(root)
store.close()
print(json.dumps(seen))
"""


def test_direct_store_and_worker_keep_everything_private_under_umask_022(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", DIRECT_ENTRY_POINTS, str(tmp_path)],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=60,
        umask=0o022,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout)
    job = seen["job"]
    database = {"data": 0o700, "data/db": 0o700, "data/db/chriek.db": 0o600}
    sidecars = {"data/db/chriek.db-wal": 0o600, "data/db/chriek.db-shm": 0o600}

    assert seen["paused"] == database | sidecars | {
        "data/images": 0o700,
        f"data/images/{job}.partial": 0o700,
        f"data/images/{job}.partial/0.png": 0o600,
    }
    published = {
        "data/images": 0o700,
        f"data/images/{job}": 0o700,
        f"data/images/{job}/0.png": 0o600,
        f"data/images/{job}/1.png": 0o600,
    }
    assert seen["done"] == database | sidecars | published
    assert seen["closed"] == database | published
    assert seen["reopened"] == database | sidecars | published  # recreated sidecars


def installed(tmp_path: Path) -> str:
    """A private installation holding one finished job; returns the job ID."""
    with running(tmp_path) as client:
        body = wait_for(client, submit(client, width=16, height=16)["id"])
    assert body["status"] == "succeeded"
    return body["id"]


def make(path: Path, mode: int, *, directory: bool = False) -> None:
    if directory:
        path.mkdir()
    else:
        path.write_bytes(b"")
    path.chmod(mode)


def relink(path: Path, target: Path) -> None:
    path.rename(target)
    path.symlink_to(target)


# How to make an existing private installation unsafe, and the path chriek must name.
UNSAFE = {
    "image root readable": (lambda t, job: (t / "images").chmod(0o755), "images"),
    "database dir readable": (lambda t, job: (t / "db").chmod(0o750), "db"),
    "database readable": (lambda t, job: (t / "db/chriek.db").chmod(0o644), "db/chriek.db"),
    "wal readable": (lambda t, job: make(t / "db/chriek.db-wal", 0o644), "db/chriek.db-wal"),
    "shm readable": (lambda t, job: make(t / "db/chriek.db-shm", 0o604), "db/chriek.db-shm"),
    "journal readable": (
        lambda t, job: make(t / "db/chriek.db-journal", 0o640),
        "db/chriek.db-journal",
    ),
    "job dir readable": (lambda t, job: (t / "images" / job).chmod(0o755), "images/{job}"),
    "image readable": (
        lambda t, job: (t / "images" / job / "0.png").chmod(0o644),
        "images/{job}/0.png",
    ),
    "partial dir traversable": (
        lambda t, job: make(t / "images" / f"{job}.partial", 0o711, directory=True),
        "images/{job}.partial",
    ),
    "image root linked": (lambda t, job: relink(t / "images", t / "real-images"), "images"),
    "database dir linked": (lambda t, job: relink(t / "db", t / "real-db"), "db"),
    "database linked": (
        lambda t, job: relink(t / "db/chriek.db", t / "db/real.db"),
        "db/chriek.db",
    ),
    "image linked": (
        lambda t, job: relink(t / "images" / job / "0.png", t / "elsewhere.png"),
        "images/{job}/0.png",
    ),
}


@pytest.mark.parametrize(("make_unsafe", "culprit"), UNSAFE.values(), ids=UNSAFE.keys())
def test_unsafe_existing_storage_is_refused_before_anything_changes(tmp_path, make_unsafe, culprit):
    job = installed(tmp_path)
    make_unsafe(tmp_path, job)
    before = tree_state(tmp_path)

    with (
        pytest.raises(StorageError, match="never changes existing permissions") as error,
        running(tmp_path),
    ):
        pass

    assert str(error.value).startswith(f"{tmp_path / culprit.format(job=job)} ")
    assert tree_state(tmp_path) == before, "refused storage must stay byte-for-byte unchanged"


def test_store_and_worker_refuse_unsafe_storage_when_used_directly(tmp_path):
    installed(tmp_path)
    (tmp_path / "db/chriek.db").chmod(0o644)
    with pytest.raises(StorageError, match=r"chriek\.db is accessible to other users \(mode 0644"):
        JobStore(tmp_path / "db/chriek.db")

    (tmp_path / "db/chriek.db").chmod(0o600)
    (tmp_path / "images").chmod(0o755)
    store = JobStore(tmp_path / "db/chriek.db")
    try:
        with pytest.raises(StorageError, match=r"images is accessible to other users"):
            Worker(store, FakeBackend(), tmp_path / "images", lambda: 0.0)
    finally:
        store.close()


def test_storage_owned_by_another_user_is_refused(tmp_path, monkeypatch):
    installed(tmp_path)
    before = tree_state(tmp_path)
    monkeypatch.setattr(os, "geteuid", lambda: os.getuid() + 1)

    with pytest.raises(StorageError, match=rf"owned by uid {os.getuid()}"), running(tmp_path):
        pass
    assert tree_state(tmp_path) == before


def shared_install(tmp_path: Path, mode: int) -> tuple[Path, dict[str, str]]:
    """A caller-owned shared directory with dedicated chriek directories configured inside."""
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "notes.txt").write_text("keep me")
    shared.chmod(mode)
    env = {
        "CHRIEK_DB_PATH": str(shared / "chriek" / "db" / "chriek.db"),
        "CHRIEK_STORAGE_DIR": str(shared / "chriek" / "images"),
    }
    return shared, env


@pytest.mark.parametrize("mode", [0o755, 0o1777], ids=["readable", "sticky world-writable"])
def test_dedicated_directories_under_a_shared_ancestor_leave_it_untouched(tmp_path, mode):
    shared, env = shared_install(tmp_path, mode)
    with running(tmp_path, **env) as client:
        job = wait_for(client, submit(client, width=16, height=16)["id"])
    assert job["status"] == "succeeded"

    assert stat.S_IMODE(shared.stat().st_mode) == mode
    assert (shared / "notes.txt").read_text() == "keep me"
    assert modes(shared / "chriek") == {
        "db": 0o700,
        "db/chriek.db": 0o600,
        "images": 0o700,
        f"images/{job['id']}": 0o700,
        f"images/{job['id']}/0.png": 0o600,
    }


@pytest.mark.parametrize("mode", [0o777, 0o775], ids=["world-writable", "group-writable"])
def test_an_ancestor_other_users_can_modify_is_refused_without_creating_anything(tmp_path, mode):
    shared, env = shared_install(tmp_path, mode)
    before = tree_state(shared)

    with (
        pytest.raises(StorageError, match="is an ancestor that other users can modify"),
        running(tmp_path, **env),
    ):
        pass
    assert tree_state(shared) == before
