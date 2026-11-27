"""scripts/check_cuda_image.sh touches only staging paths it created, in a disposable checkout."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_cuda_image.sh"
STAGING = [
    "chriek-decoy.env",
    "chriek-decoy-hf",
    "chriek-decoy-weights",
    "src/chriek/chriek-decoy.env",
    "src/chriek/chriek_decoy",
]
DECOYS = [
    "chriek-decoy.env",
    "chriek-decoy-hf/token",
    "chriek-decoy-weights/model.safetensors",
    "src/chriek/chriek-decoy.env",
    "src/chriek/chriek_decoy/model.safetensors",
]


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "src" / "chriek").mkdir(parents=True)
    (root / "src" / "chriek" / "__init__.py").write_text("dirty = True\n")
    shutil.copy(SCRIPT, root / "scripts" / "check_cuda_image.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Fake docker: records which decoys were in the build context, then fails the build.
    docker = bin_dir / "docker"
    docker.write_text(
        f'#!/bin/sh\nls {" ".join(DECOYS)} > "{tmp_path}/docker-called" 2>&1\nexit 1\n'
    )
    docker.chmod(0o755)
    return root


def run(checkout):
    env = {**os.environ, "PATH": f"{checkout.parent / 'bin'}:{os.environ['PATH']}"}
    return subprocess.run(
        ["/bin/sh", str(checkout / "scripts" / "check_cuda_image.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def plant(checkout, path, kind):
    target = checkout / path
    if kind == "empty dir":
        target.mkdir()
    else:
        target.symlink_to(checkout.parent / "operator" / "token")


@pytest.mark.parametrize("kind", ["empty dir", "dangling link"])
@pytest.mark.parametrize("path", STAGING)
def test_any_single_collision_refuses_before_writing_or_docker(checkout, path, kind):
    plant(checkout, path, kind)
    result = run(checkout)
    assert result.returncode == 1
    assert f"refusing to overwrite existing {path}" in result.stderr
    assert not (checkout.parent / "docker-called").exists()
    assert not (checkout.parent / "operator").exists()
    planted = checkout / path
    if kind == "empty dir":
        assert planted.is_dir() and not any(planted.iterdir())
    else:
        assert planted.is_symlink() and not planted.exists()
    for other in STAGING:
        if other != path:
            assert not (checkout / other).exists() and not (checkout / other).is_symlink()


def test_every_collision_is_reported_and_all_are_kept(checkout):
    plant(checkout, "chriek-decoy-hf", "empty dir")
    plant(checkout, "chriek-decoy.env", "dangling link")
    result = run(checkout)
    assert result.returncode == 1
    assert "refusing to overwrite existing chriek-decoy.env" in result.stderr
    assert "refusing to overwrite existing chriek-decoy-hf" in result.stderr
    assert (checkout / "chriek-decoy-hf").is_dir()
    assert (checkout / "chriek-decoy.env").is_symlink()
    assert not (checkout.parent / "operator").exists()
    assert not (checkout.parent / "docker-called").exists()


def test_docker_failure_removes_only_this_runs_paths(checkout):
    result = run(checkout)
    assert result.returncode != 0
    # Every decoy was in the build context when docker ran.
    assert sorted((checkout.parent / "docker-called").read_text().split()) == sorted(DECOYS)
    for path in STAGING:
        assert not (checkout / path).exists() and not (checkout / path).is_symlink()
    assert (checkout / "src" / "chriek" / "__init__.py").read_text() == "dirty = True\n"
