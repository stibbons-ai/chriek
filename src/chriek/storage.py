"""Private on-disk storage: dedicated service-owned directories (0700) and files (0600).

Existing paths are checked, never repaired: chriek refuses to start rather than chmod data.
"""

import os
import stat
from pathlib import Path
from typing import NoReturn

_PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700
_SQLITE_SIDECARS = ("-wal", "-shm", "-journal")
_GUIDANCE = (
    "chriek needs dedicated directories owned by the service user with no group or other "
    "access, under ancestors only root or the service user can modify. chriek never changes "
    "existing permissions: stop chriek, move the data into a new private directory (see "
    "README, Storage permissions) or configure a new dedicated path, then restart"
)


class StorageError(RuntimeError):
    """Configured storage is not private to the service user; nothing existing was changed."""


def private_dir(path: Path) -> Path:
    """Check or create a dedicated private directory and return its real path.

    Missing levels are created 0700; existing ancestors are only checked, never modified.
    """
    path = Path(os.path.abspath(path))
    if path.is_symlink():
        _reject(path, "is a symbolic link")
    real = path.resolve()
    missing = [real]
    for parent in real.parents:
        if not parent.exists():
            missing.append(parent)
            continue
        info = os.stat(parent)
        if info.st_uid not in (0, os.geteuid()):
            _reject(parent, f"is an ancestor owned by uid {info.st_uid}")
        if info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX:
            _reject(parent, "is an ancestor that other users can modify")
    for directory in reversed(missing):
        if not directory.exists():
            os.mkdir(directory, PRIVATE_DIR_MODE)
    if not stat.S_ISDIR(_check_private(real).st_mode):
        _reject(real, "is not a directory")
    return real


def private_image_root(path: Path) -> Path:
    """`private_dir` plus a check of every stored entry below it."""
    root = private_dir(path)
    for entry in root.rglob("*"):  # yields links without following them
        _check_private(entry)
    return root


def private_database(db_path: Path) -> Path:
    """Check the database and its SQLite sidecars; create a missing database file 0600.

    SQLite creates its sidecars with the database file's mode, so they stay private too.
    """
    db_path = private_dir(db_path.parent) / db_path.name
    for path in (db_path, *(Path(f"{db_path}{suffix}") for suffix in _SQLITE_SIDECARS)):
        if path.exists(follow_symlinks=False) and not stat.S_ISREG(_check_private(path).st_mode):
            _reject(path, "is not a regular file")
    if not db_path.exists():
        write_private(db_path, b"")
    return db_path


def write_private(path: Path, data: bytes) -> None:
    """Create a new file readable only by the service user from its first byte."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _PRIVATE_FILE_MODE)
    with open(fd, "wb") as file:
        file.write(data)


def _check_private(path: Path) -> os.stat_result:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode):
        _reject(path, "is a symbolic link")
    if info.st_uid != os.geteuid():
        _reject(path, f"is owned by uid {info.st_uid}, not the service user")
    if info.st_mode & 0o077:
        _reject(path, f"is accessible to other users (mode {stat.S_IMODE(info.st_mode):04o})")
    return info


def _reject(path: Path, problem: str) -> NoReturn:
    raise StorageError(f"{path} {problem}. {_GUIDANCE}")
