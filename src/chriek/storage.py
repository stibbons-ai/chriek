"""Private on-disk storage: dedicated service-owned directories (0700) and files (0600).

Existing paths are checked, never repaired: chriek refuses to start rather than chmod data.
"""

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn

_PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700
_SQLITE_SIDECARS = ("-wal", "-shm", "-journal")
_GUIDANCE = (
    "chriek needs dedicated directories owned by the service user with no group or other "
    "access, reached only through directories and links root or the service user controls. "
    "chriek never changes existing permissions: stop chriek, then remove group and other access "
    "from chriek's own data (chmod -R go= <dedicated dir>, never a shared parent), or move the "
    "data into a new dedicated directory, remove its group and other access and point "
    "CHRIEK_DB_PATH or CHRIEK_STORAGE_DIR there; restart (see README, Storage permissions)"
)


class StorageError(RuntimeError):
    """Configured storage is unsafe, or can't be checked or set up; nothing existing was changed."""


def private_image_root(path: Path) -> Path:
    """Check the image root and everything below it; create it 0700 if missing; return real path."""
    with _os_errors(path):
        root, missing = _checked_dir(path)
        if not missing:
            _check_tree(root)
        _create(missing)
    return root


def check_database(db_path: Path) -> None:
    """Every check `private_database` makes, without creating anything."""
    with _os_errors(db_path):
        _checked_database(db_path)


def private_database(db_path: Path) -> Path:
    """Check the database and its SQLite sidecars; create a missing database file 0600.

    SQLite creates its sidecars with the database file's mode, so they stay private too.
    """
    with _os_errors(db_path):
        db_path, missing = _checked_database(db_path)
        _create(missing)
        if not db_path.exists():
            write_private(db_path, b"")
    return db_path


def write_private(path: Path, data: bytes) -> None:
    """Create a new file readable only by the service user from its first byte."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _PRIVATE_FILE_MODE)
    with open(fd, "wb") as file:
        file.write(data)


def _checked_database(db_path: Path) -> tuple[Path, list[Path]]:
    parent, missing = _checked_dir(db_path.parent)
    db_path = parent / db_path.name
    for path in (db_path, *(Path(f"{db_path}{suffix}") for suffix in _SQLITE_SIDECARS)):
        if _stat(path) and not stat.S_ISREG(_check_private(path).st_mode):
            _reject(path, "is not a regular file")
    return db_path, missing


def _checked_dir(path: Path) -> tuple[Path, list[Path]]:
    """Check a dedicated directory and its ancestors without creating anything.

    Returns the real path and its missing levels, outermost first.
    """
    path = Path(os.path.abspath(path))
    parent, missing = _walk(path.parent)
    real = parent / path.name
    info = None if missing else _stat(real)
    if info is None:
        missing.append(real)
    elif stat.S_ISLNK(info.st_mode):
        _reject(path, "is a symbolic link")
    elif not stat.S_ISDIR(_check_private(real).st_mode):
        _reject(real, "is not a directory")
    return real, missing


def _walk(path: Path) -> tuple[Path, list[Path]]:
    """Resolve absolute `path` like the kernel, checking every directory and link it follows.

    Returns the real path and its missing levels, outermost first. This is the only walk that
    follows links: missing levels are later created with mkdir, which refuses anything that
    appeared meanwhile, so no link is ever followed unchecked. A link another user owns, or one
    in a directory they can modify, could choose where chriek's directories end up.
    """
    pending = list(reversed(path.parts[1:]))
    current, links = Path(path.anchor), 0
    while (info := _stat(current)) is not None:
        if stat.S_ISLNK(info.st_mode):
            # The directory holding the link was checked on the way in.
            if info.st_uid not in (0, os.geteuid()):
                _reject(current, f"is a symbolic link owned by uid {info.st_uid}")
            links += 1
            if links > 40:
                raise OSError(errno.ELOOP, os.strerror(errno.ELOOP), str(current))
            target = Path(os.readlink(current))
            pending.extend(reversed(target.parts[1:] if target.is_absolute() else target.parts))
            current = Path(target.anchor) if target.is_absolute() else current.parent
            continue
        _check_ancestor(current, info)
        if not pending:
            return current, []
        part = pending.pop()
        current = current.parent if part == ".." else current / part
    missing = [current]
    for part in reversed(pending):
        if part == "..":
            # The kernel can't resolve `<missing>/..`; guessing where it leads would be unchecked.
            _reject(missing[-1], "is missing, so the link target leading through it can't be used")
        missing.append(missing[-1] / part)
    return missing[-1], missing


def _check_ancestor(path: Path, info: os.stat_result) -> None:
    if not stat.S_ISDIR(info.st_mode):
        _reject(path, "is an ancestor that is not a directory")
    if info.st_uid not in (0, os.geteuid()):
        _reject(path, f"is an ancestor owned by uid {info.st_uid}")
    if info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX:
        _reject(path, "is an ancestor that other users can modify")


def _stat(path: Path) -> os.stat_result | None:
    """lstat, or None only if `path` is missing; every other error (no access, a path through a
    link loop) raises."""
    try:
        return os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _create(missing: list[Path]) -> None:
    for directory in missing:
        try:
            os.mkdir(directory, PRIVATE_DIR_MODE)
        except FileExistsError:
            # Created by someone else after the checks: never adopt or create below it.
            _reject(directory, "appeared while chriek was creating it")


def _check_tree(root: Path) -> None:
    def fail(error: OSError) -> NoReturn:
        raise error

    # os.walk doesn't follow links; `fail` stops it skipping directories it can't list.
    for directory, dirs, files in os.walk(root, onerror=fail):
        for name in dirs + files:
            _check_private(Path(directory, name))


def _check_private(path: Path) -> os.stat_result:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode):
        _reject(path, "is a symbolic link")
    if info.st_uid != os.geteuid():
        _reject(path, f"is owned by uid {info.st_uid}, not the service user")
    if info.st_mode & 0o077:
        _reject(path, f"is accessible to other users (mode {stat.S_IMODE(info.st_mode):04o})")
    return info


@contextmanager
def _os_errors(path: Path) -> Iterator[None]:
    try:
        yield
    except OSError as error:
        culprit = error.filename or path
        problem = error.strerror or error
        raise StorageError(f"{culprit} cannot be used ({problem}). {_GUIDANCE}") from error


def _reject(path: Path, problem: str) -> NoReturn:
    raise StorageError(f"{path} {problem}. {_GUIDANCE}")
