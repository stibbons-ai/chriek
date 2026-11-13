"""Storage privacy at the store, worker and app boundaries, without the CLI."""

import io
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest
from conftest import PausingBackend, modes, running, submit, tree_state, wait_for
from PIL import Image

from chriek.backend import FakeBackend
from chriek.storage import StorageError, private_image_root
from chriek.store import JobStore
from chriek.worker import Worker

# Runs in a child process so its umask never leaks into other tests.
DIRECT_ENTRY_POINTS = """
import json, sys, time
from pathlib import Path

from conftest import PausingBackend, modes
from chriek.store import SUCCEEDED, JobStore
from chriek.worker import Worker

root = Path(sys.argv[1])
db_path = root / "data" / "db" / "chriek.db"
exposed = set()


def watch(event, args):
    # Before every Python-level open, chmod, mkdir or rename, nothing created so far may be
    # open to other users: a file created readable and fixed later is caught at the fix.
    if event not in ("open", "os.chmod", "os.mkdir", "os.rename"):
        return
    for path in root.rglob("*"):
        try:
            if path.lstat().st_mode & 0o077:
                exposed.add(str(path.relative_to(root)))
        except FileNotFoundError:  # removed by the worker meanwhile
            pass


sys.addaudithook(watch)

store = JobStore(db_path)
job = store.create(owner="alice", prompt="p", image_count=2, width=16, height=16, seed=0, now=0.0)
backend = PausingBackend()
worker = Worker(store, backend, root / "data" / "images", time.time)
worker.start()
assert backend.paused.wait(10)
seen = {"job": job.id, "paused": modes(root)}
backend.resume.set()
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
print(json.dumps(seen | {"exposed": sorted(exposed)}))
"""


@pytest.mark.parametrize("umask", [0o000, 0o022], ids=["umask 000", "umask 022"])
def test_direct_store_and_worker_keep_everything_private_whatever_the_umask(tmp_path, umask):
    result = subprocess.run(
        [sys.executable, "-c", DIRECT_ENTRY_POINTS, str(tmp_path)],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=60,
        umask=umask,
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
    assert seen["exposed"] == [], "created readable, made private only afterwards"
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


def relink(path: Path, target: Path) -> str:
    """Replace `path` by a link to `target`; returns the refusal reason chriek must give."""
    path.rename(target)
    path.symlink_to(target)
    return "is a symbolic link."


def replace(path: Path, mode: int, *, directory: bool = False) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    make(path, mode, directory=directory)


# How to make an existing private installation unsafe, and the path chriek must name.
UNSAFE = {
    "image root readable": (lambda t, job: (t / "images").chmod(0o755), "images"),
    "database dir readable": (lambda t, job: (t / "db").chmod(0o750), "db"),
    "image root traversable by others": (lambda t, job: (t / "images").chmod(0o701), "images"),
    "image root traversable by group": (lambda t, job: (t / "images").chmod(0o710), "images"),
    "database dir traversable by others": (lambda t, job: (t / "db").chmod(0o701), "db"),
    "database dir traversable by group": (lambda t, job: (t / "db").chmod(0o710), "db"),
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
    "database group-writable": (lambda t, job: (t / "db/chriek.db").chmod(0o620), "db/chriek.db"),
    "database other-writable": (lambda t, job: (t / "db/chriek.db").chmod(0o602), "db/chriek.db"),
    "partial dir traversable": (
        lambda t, job: make(t / "images" / f"{job}.partial", 0o711, directory=True),
        "images/{job}.partial",
    ),
    "image group-writable": (
        lambda t, job: (t / "images" / job / "0.png").chmod(0o620),
        "images/{job}/0.png",
    ),
    "image other-writable": (
        lambda t, job: (t / "images" / job / "0.png").chmod(0o602),
        "images/{job}/0.png",
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
    "image root not a directory": (lambda t, job: replace(t / "images", 0o600), "images"),
    "database dir not a directory": (lambda t, job: replace(t / "db", 0o600), "db"),
    "database not a regular file": (
        lambda t, job: replace(t / "db/chriek.db", 0o700, directory=True),
        "db/chriek.db",
    ),
    "wal not a regular file": (
        lambda t, job: make(t / "db/chriek.db-wal", 0o700, directory=True),
        "db/chriek.db-wal",
    ),
}


@pytest.mark.parametrize(("make_unsafe", "culprit"), UNSAFE.values(), ids=UNSAFE.keys())
def test_unsafe_existing_storage_is_refused_before_anything_changes(tmp_path, make_unsafe, culprit):
    job = installed(tmp_path)
    problem = make_unsafe(tmp_path, job) or ""  # links must be refused as links, whatever mode
    before = tree_state(tmp_path)

    with (
        pytest.raises(StorageError, match="never changes existing permissions") as error,
        running(tmp_path),
    ):
        pass

    assert str(error.value).startswith(f"{tmp_path / culprit.format(job=job)} {problem}")
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


def report_owned_by_another_user(monkeypatch, target: Path, uid: int = os.geteuid() + 1) -> int:
    """Make stat and lstat report `target` as owned by `uid`; returns that UID."""
    for name in ("stat", "lstat"):
        real = getattr(os, name)

        def fake(path, *args, _real=real, **kwargs):
            info = _real(path, *args, **kwargs)
            if isinstance(path, str | Path) and Path(path) == target:
                return os.stat_result((*info[:4], uid, *info[5:10]))
            return info

        monkeypatch.setattr(os, name, fake)
    return uid


@pytest.mark.parametrize("install", ["", "mine"], ids=["parent", "above a private parent"])
def test_an_ancestor_owned_by_another_user_is_refused(tmp_path, monkeypatch, install):
    installed(tmp_path / install)
    before = tree_state(tmp_path)
    foreign = report_owned_by_another_user(monkeypatch, tmp_path)

    with pytest.raises(StorageError) as error, running(tmp_path / install):
        pass
    monkeypatch.undo()
    assert str(error.value).startswith(f"{tmp_path} is an ancestor owned by uid {foreign}.")
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize(
    "culprit",
    ["images", "images/{job}", "images/{job}/0.png", "db", "db/chriek.db", "db/chriek.db-wal"],
)
def test_storage_owned_by_another_user_is_refused(tmp_path, monkeypatch, culprit):
    job = installed(tmp_path)
    make(tmp_path / "db/chriek.db-wal", 0o600)
    culprit = tmp_path / culprit.format(job=job)
    before = tree_state(tmp_path)
    foreign = report_owned_by_another_user(monkeypatch, culprit)

    with pytest.raises(StorageError) as error, running(tmp_path):
        pass
    monkeypatch.undo()
    assert str(error.value).startswith(f"{culprit} is owned by uid {foreign}, not the service")
    assert tree_state(tmp_path) == before


# Each case: an unsafe side plus a missing side; neither may be created before the refusal.
PARTLY_MISSING = {
    "unsafe database root, images missing inside it": (
        lambda t: (make(t / "data", 0o755, directory=True), make(t / "data/chriek.db", 0o644)),
        {"CHRIEK_DB_PATH": "data/chriek.db", "CHRIEK_STORAGE_DIR": "data/images"},
    ),
    "unsafe database root, images missing elsewhere": (
        lambda t: make(t / "db", 0o755, directory=True),
        {"CHRIEK_DB_PATH": "db/chriek.db", "CHRIEK_STORAGE_DIR": "new/images"},
    ),
    "unsafe image root, database missing": (
        lambda t: make(t / "images", 0o755, directory=True),
        {"CHRIEK_DB_PATH": "new/chriek.db", "CHRIEK_STORAGE_DIR": "images"},
    ),
}


@pytest.mark.parametrize(("setup", "paths"), PARTLY_MISSING.values(), ids=PARTLY_MISSING.keys())
def test_one_unsafe_side_stops_startup_before_the_other_is_created(tmp_path, setup, paths):
    setup(tmp_path)
    before = tree_state(tmp_path)
    env = {name: str(tmp_path / path) for name, path in paths.items()}

    with pytest.raises(StorageError), running(tmp_path, **env):
        pass
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize("inject", ["directory", "symlink"])
def test_a_level_appearing_while_chriek_creates_it_is_refused(tmp_path, monkeypatch, inject):
    """Stand-in for another user racing chriek in a sticky world-writable directory."""
    sticky = tmp_path / "sticky"
    sticky.mkdir()
    sticky.chmod(0o1777)
    (tmp_path / "elsewhere").mkdir(mode=0o700)
    level = sticky / "chriek-data"
    real_stat = os.stat
    injected = {}

    def racing_stat(path, *args, **kwargs):
        # The journal sidecar is the last path checked: inject after every check, before any
        # creation.
        if not injected and str(path).endswith("-journal"):
            if inject == "directory":
                os.mkdir(level, 0o777)
                level.chmod(0o777)
            else:
                level.symlink_to(tmp_path / "elsewhere")
            injected.update(tree_state(tmp_path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", racing_stat)
    with pytest.raises(StorageError) as error:
        JobStore(level / "db" / "chriek.db")
    monkeypatch.undo()
    assert str(error.value).startswith(f"{level} ")
    assert tree_state(tmp_path) == injected, "nothing may be created in or below the newcomer"


@pytest.mark.parametrize("side", ["images", "db"])
def test_a_configured_link_to_a_target_below_a_missing_directory_is_refused(tmp_path, side):
    (tmp_path / side).symlink_to(tmp_path / "elsewhere" / "deep")  # `elsewhere` doesn't exist
    before = tree_state(tmp_path)

    with pytest.raises(StorageError) as error, running(tmp_path):
        pass
    assert str(error.value).startswith(f"{tmp_path / side} is a symbolic link.")
    assert tree_state(tmp_path) == before, "nothing may be created at the target or other side"


@pytest.mark.parametrize("side", ["images", "db"])
def test_a_missing_level_linked_right_after_chriek_sees_it_missing_is_refused(
    tmp_path, monkeypatch, side
):
    """Stand-in for another user filling a missing level in a sticky directory with a link."""
    sticky = tmp_path / "sticky"
    make(sticky, 0o1777, directory=True)
    (tmp_path / "target").mkdir(mode=0o700)
    level = sticky / "chriek-data"
    real_stat = os.stat
    injected = {}

    def racing_stat(path, *args, **kwargs):
        try:
            return real_stat(path, *args, **kwargs)
        except FileNotFoundError:
            if not injected and Path(path) == level:
                level.symlink_to(tmp_path / "target")
                injected.update(tree_state(tmp_path))
            raise

    monkeypatch.setattr(os, "stat", racing_stat)
    with pytest.raises(StorageError) as error:
        if side == "db":
            JobStore(level / "db" / "chriek.db")
        else:
            private_image_root(level / "images")
    monkeypatch.undo()
    assert injected, "the link was never planted"
    assert str(error.value).startswith(f"{level} appeared while chriek was creating it.")
    assert tree_state(tmp_path) == injected, "nothing may be created through the link"


@pytest.mark.parametrize("side", ["images", "db"])
def test_a_trusted_link_through_a_missing_level_and_dotdot_is_refused(tmp_path, side):
    """`ghost/..` can't be resolved by the kernel, so where it leads was never checked."""
    (tmp_path / "chosen").mkdir(mode=0o700)
    (tmp_path / "evil").symlink_to(tmp_path / "chosen")
    (tmp_path / "link").symlink_to("ghost/../evil")
    before = tree_state(tmp_path)

    with pytest.raises(StorageError) as error:
        if side == "db":
            JobStore(tmp_path / "link" / "db" / "chriek.db")
        else:
            private_image_root(tmp_path / "link" / "images")
    assert str(error.value).startswith(f"{tmp_path / 'ghost'} is missing")
    assert tree_state(tmp_path) == before, "nothing may be created at the link's target"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every directory")
def test_an_image_directory_chriek_cannot_list_is_refused(tmp_path):
    job = installed(tmp_path)
    (tmp_path / "images" / job / "0.png").chmod(0o644)
    before = tree_state(tmp_path)
    (tmp_path / "images" / job).chmod(0o300)  # entries can't be listed, so not checked

    try:
        with pytest.raises(StorageError) as error, running(tmp_path):
            pass
    finally:
        (tmp_path / "images" / job).chmod(0o700)
    assert str(error.value).startswith(f"{tmp_path / 'images' / job} ")
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize("blocked", ["looped", "unsearchable"])
def test_a_database_path_chriek_cannot_inspect_is_refused_before_anything_is_created(
    tmp_path, blocked
):
    if blocked == "looped":
        (tmp_path / blocked).symlink_to(blocked)
    elif os.geteuid() == 0:
        pytest.skip("root searches every directory")
    else:
        (tmp_path / blocked).mkdir(mode=0o600)  # private, but chriek can't look inside
    env = {
        "CHRIEK_DB_PATH": str(tmp_path / blocked / "db" / "chriek.db"),
        "CHRIEK_STORAGE_DIR": str(tmp_path / "images"),
    }
    before = tree_state(tmp_path)

    with pytest.raises(StorageError) as error, running(tmp_path, **env):
        pass
    assert str(error.value).startswith(f"{tmp_path / blocked}")
    assert "appeared while chriek was creating it" not in str(error.value)
    assert tree_state(tmp_path) == before, "images must not be created before the refusal"


@pytest.mark.parametrize(
    "target",
    ["{tmp}/real", "real", "real/../real"],
    ids=["absolute", "relative", "relative dotdot"],
)
def test_a_linked_ancestor_is_followed_to_its_private_target(tmp_path, target):
    """Like macOS's /var -> /private/var: links above the configured paths are fine."""
    (tmp_path / "real").mkdir(mode=0o700)
    (tmp_path / "alias").symlink_to(target.format(tmp=tmp_path))
    env = {
        "CHRIEK_DB_PATH": str(tmp_path / "alias/chriek/db/chriek.db"),
        "CHRIEK_STORAGE_DIR": str(tmp_path / "alias/chriek/images"),
    }
    with running(tmp_path, **env) as client:
        job = wait_for(client, submit(client, width=16, height=16)["id"])
        assert client.get(job["images"][0]["url"]).status_code == 200

    assert modes(tmp_path / "real") == {
        "chriek": 0o700,
        "chriek/db": 0o700,
        "chriek/db/chriek.db": 0o600,
        "chriek/images": 0o700,
        f"chriek/images/{job['id']}": 0o700,
        f"chriek/images/{job['id']}/0.png": 0o600,
    }


CANONICAL_PATHS = """
import sys, time
from pathlib import Path

from chriek.backend import FakeBackend
from chriek.store import SUCCEEDED, JobStore
from chriek.worker import Worker

root = Path(sys.argv[1])
store = JobStore(root / "alias" / ".." / "db" / "chriek.db")
job = store.create(owner="alice", prompt="p", image_count=1, width=16, height=16, seed=0, now=0.0)
worker = Worker(store, FakeBackend(), root / "alias" / ".." / "images", time.time)
worker.start()
deadline = time.monotonic() + 10
try:
    while store.get("alice", job.id).status != SUCCEEDED:
        assert time.monotonic() < deadline, "job did not finish"
        time.sleep(0.02)
finally:
    worker.stop(10)
    store.close()
print(job.id)
"""


def test_paths_through_a_linked_level_and_dotdot_are_used_where_they_point(tmp_path):
    """`alias/../x` means the `x` beside `alias`, as documented, never one under its target."""
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    for directory in (real / "sub", real / "db", real / "images"):
        make(directory, 0o777, directory=True)  # unchecked shared directories chriek must ignore
    (tmp_path / "alias").symlink_to(real / "sub")
    before = tree_state(real)

    result = subprocess.run(
        [sys.executable, "-c", CANONICAL_PATHS, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        umask=0o022,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    job = result.stdout.strip()

    assert {name: modes(tmp_path)[name] for name in ("db", "images")} == {
        "db": 0o700,
        "images": 0o700,
    }
    assert modes(tmp_path / "db") == {"chriek.db": 0o600}
    assert modes(tmp_path / "images") == {job: 0o700, f"{job}/0.png": 0o600}
    assert tree_state(real) == before, "nothing may be created or changed below the link target"


def test_the_app_serves_images_from_the_checked_root_of_a_linked_dotdot_path(tmp_path):
    """Downloads read the same canonical `alias/../images` the worker wrote, never the target's."""
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    for directory in (real / "sub", real / "images"):
        make(directory, 0o777, directory=True)  # unchecked shared directories chriek must ignore
    (tmp_path / "alias").symlink_to(real / "sub")
    before = tree_state(real)

    with running(tmp_path, CHRIEK_STORAGE_DIR=str(tmp_path / "alias/../images")) as client:
        job = wait_for(client, submit(client, width=16, height=16)["id"])
        assert job["status"] == "succeeded"
        assert modes(tmp_path / "images") == {job["id"]: 0o700, f"{job['id']}/0.png": 0o600}
        assert tree_state(real) == before, "nothing may be created or changed below the link target"
        decoy = real / "images" / job["id"] / "0.png"
        decoy.parent.mkdir()
        decoy.write_bytes(b"planted bytes from an unchecked directory")

        response = client.get(job["images"][0]["url"])

    assert response.status_code == 200
    assert response.content == (tmp_path / "images" / job["id"] / "0.png").read_bytes()
    assert response.content.startswith(b"\x89PNG")


def shared_install(tmp_path: Path, mode: int, below: str = "") -> tuple[Path, dict[str, str]]:
    """A caller-owned shared directory with dedicated chriek directories configured inside.

    `below` names a private directory between the shared one and chriek's directories.
    """
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "notes.txt").write_text("keep me")
    if below:
        (shared / below).mkdir(mode=0o700)
    shared.chmod(mode)
    env = {
        "CHRIEK_DB_PATH": str(shared / below / "chriek" / "db" / "chriek.db"),
        "CHRIEK_STORAGE_DIR": str(shared / below / "chriek" / "images"),
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


@pytest.mark.parametrize("below", ["", "mine"], ids=["parent", "above a private parent"])
@pytest.mark.parametrize(
    "mode", [0o777, 0o775, 0o757], ids=["world-writable", "group-writable", "other-writable"]
)
def test_an_ancestor_other_users_can_modify_is_refused_without_creating_anything(
    tmp_path, mode, below
):
    shared, env = shared_install(tmp_path, mode, below)
    before = tree_state(shared)

    with pytest.raises(StorageError) as error, running(tmp_path, **env):
        pass
    assert str(error.value).startswith(f"{shared} is an ancestor that other users can modify.")
    assert tree_state(shared) == before


def test_the_app_keeps_a_paused_job_private_and_serves_it_once_done(tmp_path):
    backend = PausingBackend()
    with running(tmp_path, backend=backend) as client:
        job = submit(client, image_count=2, seed=0, width=16, height=16)["id"]
        try:
            assert backend.paused.wait(10), "the second image never started"
            paused = modes(tmp_path)
        finally:
            backend.resume.set()
        body = wait_for(client, job)
        download = client.get(body["images"][1]["url"])

    assert paused == {
        "db": 0o700,
        "db/chriek.db": 0o600,
        "db/chriek.db-wal": 0o600,
        "db/chriek.db-shm": 0o600,
        "images": 0o700,
        f"images/{job}.partial": 0o700,
        f"images/{job}.partial/0.png": 0o600,
    }
    assert body["status"] == "succeeded"
    assert download.status_code == 200
    assert Image.open(io.BytesIO(download.content)).text["Seed"] == "1"


def public_alias(tmp_path: Path) -> Path:
    """`public/alias` -> a private directory, where `public` is world-writable and not sticky."""
    public = tmp_path / "public"
    public.mkdir()
    public.chmod(0o777)
    (tmp_path / "private").mkdir(mode=0o700)
    (public / "alias").symlink_to(tmp_path / "private")
    return public / "alias"


def chained_alias(tmp_path: Path) -> Path:
    """`entry` -> `public/alias` -> a private directory: no unsafe level in the configured path."""
    (tmp_path / "entry").symlink_to(public_alias(tmp_path))
    return tmp_path / "entry"


def relative_alias(tmp_path: Path) -> Path:
    """`entry` -> `public/alias` -> `../private`, all relative, `public` as in `public_alias`."""
    make(tmp_path / "public", 0o777, directory=True)
    (tmp_path / "private").mkdir(mode=0o700)
    (tmp_path / "public" / "alias").symlink_to("../private")
    (tmp_path / "entry").symlink_to("public/alias")
    return tmp_path / "entry"


def nested_relative_alias(tmp_path: Path) -> Path:
    """`deep/up` -> `../entry/x`, a relative chain into `relative_alias` from a lower level."""
    relative_alias(tmp_path)
    (tmp_path / "deep").mkdir(mode=0o700)
    (tmp_path / "deep" / "up").symlink_to("../entry/x")
    return tmp_path / "deep" / "up"


def unsafe_target(tmp_path: Path) -> Path:
    """`entry` -> `public`, a world-writable directory that isn't sticky."""
    make(tmp_path / "public", 0o777, directory=True)
    (tmp_path / "entry").symlink_to(tmp_path / "public")
    return tmp_path / "entry"


def relative_unsafe_target(tmp_path: Path) -> Path:
    """`entry` -> `public` as in `unsafe_target`, written relative."""
    make(tmp_path / "public", 0o777, directory=True)
    (tmp_path / "entry").symlink_to("public")
    return tmp_path / "entry"


@pytest.mark.parametrize(
    "alias",
    [
        public_alias,
        chained_alias,
        relative_alias,
        nested_relative_alias,
        unsafe_target,
        relative_unsafe_target,
    ],
    ids=["direct", "chained", "relative", "nested relative", "unsafe target", "relative unsafe"],
)
def test_a_link_in_an_ancestor_other_users_can_modify_is_refused(tmp_path, alias):
    """Anyone could re-point `public/alias` and choose where chriek's data goes."""
    env = {
        "CHRIEK_DB_PATH": str(alias(tmp_path) / "chriek" / "db" / "chriek.db"),
        "CHRIEK_STORAGE_DIR": str(tmp_path / "images"),
    }
    before = tree_state(tmp_path)

    with pytest.raises(StorageError) as error, running(tmp_path, **env):
        pass
    assert str(error.value).startswith(
        f"{tmp_path / 'public'} is an ancestor that other users can modify."
    )
    assert tree_state(tmp_path) == before


def test_a_link_another_user_owns_in_a_sticky_directory_is_refused(tmp_path, monkeypatch):
    """Stand-in (faked owner) for another user's link in /tmp choosing a directory of ours."""
    sticky = tmp_path / "sticky"
    sticky.mkdir()
    sticky.chmod(0o1777)
    (tmp_path / "home").mkdir(mode=0o700)
    link = sticky / "chriek-data"
    link.symlink_to(tmp_path / "home" / "chosen")
    before = tree_state(tmp_path)
    foreign = report_owned_by_another_user(monkeypatch, link)

    with pytest.raises(StorageError) as error:
        JobStore(link / "db" / "chriek.db")
    monkeypatch.undo()
    assert str(error.value).startswith(f"{link} is a symbolic link owned by uid {foreign}.")
    assert tree_state(tmp_path) == before, "nothing may be created at the link's target"


@pytest.mark.parametrize("owner", ["root", "service user"])
def test_a_trusted_link_in_a_sticky_directory_is_refused(tmp_path, monkeypatch, owner):
    """Anyone can hard-link a root-owned link into /tmp (macOS), so its owner proves nothing."""
    sticky = tmp_path / "sticky"
    make(sticky, 0o1777, directory=True)
    (tmp_path / "home").mkdir(mode=0o700)
    link = sticky / "chriek-data"
    link.symlink_to(tmp_path / "home" / "chosen")
    before = tree_state(tmp_path)
    if owner == "root":
        report_owned_by_another_user(monkeypatch, link, uid=0)

    with pytest.raises(StorageError) as error:
        JobStore(link / "db" / "chriek.db")
    monkeypatch.undo()
    assert str(error.value).startswith(
        f"{link} is a symbolic link in a directory other users can write to."
    )
    assert tree_state(tmp_path) == before, "nothing may be created at the link's target"


@pytest.mark.skipif(not Path("/var").is_symlink(), reason="needs macOS's root-owned /var link")
def test_a_root_owned_system_alias_is_followed(tmp_path):
    if not tmp_path.is_relative_to("/private/var"):
        pytest.skip("temporary directory is not under /private/var")
    via_var = Path("/var") / tmp_path.relative_to("/private/var")
    env = {
        "CHRIEK_DB_PATH": str(via_var / "db" / "chriek.db"),
        "CHRIEK_STORAGE_DIR": str(via_var / "images"),
    }
    with running(tmp_path, **env) as client:
        job = wait_for(client, submit(client, width=16, height=16)["id"])
        assert client.get(job["images"][0]["url"]).status_code == 200
    assert modes(tmp_path)["images"] == 0o700


def test_repointing_a_trusted_alias_after_startup_never_moves_storage(tmp_path):
    """Pins the documented contract: links above the configured paths resolve once, at startup."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(mode=0o700)
    second.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(first)
    env = {
        "CHRIEK_DB_PATH": str(alias / "chriek" / "db" / "chriek.db"),
        "CHRIEK_STORAGE_DIR": str(alias / "chriek" / "images"),
    }
    with running(tmp_path, **env) as client:
        alias.unlink()
        alias.symlink_to(second)
        body = wait_for(client, submit(client, width=16, height=16)["id"])
        assert body["status"] == "succeeded", body
        download = client.get(body["images"][0]["url"])

    assert download.status_code == 200
    assert download.content == (first / "chriek/images" / body["id"] / "0.png").read_bytes()
    assert download.content.startswith(b"\x89PNG")
    assert list(second.iterdir()) == [], "nothing may follow the re-pointed link"
    with closing(sqlite3.connect(first / "chriek/db/chriek.db")) as db:
        statuses = db.execute("SELECT status FROM jobs WHERE id = ?", (body["id"],)).fetchall()
    assert statuses == [("succeeded",)]
