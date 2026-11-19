"""The local smoke command needs backend settings, not server settings."""

import io
import json
import os

import pytest
from PIL import Image

from chriek.backend import FakeBackend
from chriek.cli import main
from chriek.png import DIGITAL_SOURCE_TYPE, XMP_KEY


@pytest.fixture(autouse=True)
def no_server_settings(monkeypatch):
    for name in os.environ:
        if name.upper().startswith("CHRIEK_"):
            monkeypatch.delenv(name)


def test_smoke_edits_with_repeated_references_and_writes_disclosed_png(tmp_path, capsys):
    reference = tmp_path / "ref.png"
    output = tmp_path / "output.png"
    with Image.new("RGB", (16, 16), "red") as image:
        image.save(reference)
    args = [
        "smoke",
        "--out",
        str(output),
        "--prompt",
        "red cat",
        "--width",
        "32",
        "--height",
        "16",
        "--seed",
        "9",
    ]
    assert main(args) == 0  # text-to-image first
    baseline = output.read_bytes()
    capsys.readouterr()

    assert main(args + ["--reference", str(reference)] * 2) == 0

    record = json.loads(capsys.readouterr().out)
    assert record.pop("render_seconds") >= 0
    assert record == {
        "backend": "fake",
        "model": "chriek-fake-backend",
        "revision": "fake-v1",
        "prompt": "red cat",
        "seed": 9,
        "steps": None,
        "width": 32,
        "height": 16,
        "references": 2,
        "output": str(output),
    }
    assert output.read_bytes() != baseline
    with Image.open(output) as image:
        assert (image.format, image.mode, image.size) == ("PNG", "RGB", (32, 16))
        assert image.info["AI-Generated"] == "true"
        assert image.info["Model"] == "chriek-fake-backend"
        assert image.info["Model-Revision"] == "fake-v1"
        assert image.info["Seed"] == "9"
        assert DIGITAL_SOURCE_TYPE in image.info[XMP_KEY]


def test_smoke_keeps_in_place_reference_when_png_write_fails(tmp_path, monkeypatch, capsys):
    output = tmp_path / "reference.png"
    with Image.new("RGB", (16, 16), "red") as image:
        image.save(output)
    original = output.read_bytes()
    real_open = io.open

    def open_with_failed_write(path, mode="r", *args, **kwargs):
        file = real_open(path, mode, *args, **kwargs)
        if "w" in mode and "b" in mode:
            write = file.write

            def fail(data):
                write(data[:16])
                raise OSError("disk full")

            file.write = fail
        return file

    monkeypatch.setattr(io, "open", open_with_failed_write)
    args = ["smoke", "--out", str(output), "--reference", str(output)]
    args += ["--width", "16", "--height", "16"]
    assert main(args) == 1
    assert "cannot write PNG" in capsys.readouterr().err
    assert output.read_bytes() == original
    assert list(tmp_path.iterdir()) == [output]


def test_smoke_rejects_sequential_offload(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHRIEK_OFFLOAD", "sequential")
    output = tmp_path / "output.png"
    assert main(["smoke", "--out", str(output), "--width", "16", "--height", "16"]) == 2
    assert "CHRIEK_OFFLOAD must" in capsys.readouterr().err
    assert not output.exists()


def test_smoke_rejects_unknown_chriek_settings(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHRIEK_BACKNED", "fake")
    output = tmp_path / "output.png"
    assert main(["smoke", "--out", str(output)]) == 2
    assert "unknown setting CHRIEK_BACKNED" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(("flag", "value"), [("--width", "17"), ("--height", "0")])
def test_smoke_rejects_dimensions_outside_positive_16_pixel_grid(tmp_path, capsys, flag, value):
    output = tmp_path / "output.png"
    with pytest.raises(SystemExit) as error:
        main(["smoke", "--out", str(output), flag, value])
    assert error.value.code == 2
    assert "multiple of 16" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(
    ("width", "height", "message"),
    [
        ("32768", "32768", "--width must be"),
        ("16", "32768", "--height must be"),
        ("2048", "1024", "width * height must be at most 1048576 pixels"),
    ],
)
def test_smoke_rejects_oversized_renders(tmp_path, monkeypatch, capsys, width, height, message):
    def unexpected_render(*args, **kwargs):
        pytest.fail("oversized render reached the backend")

    monkeypatch.setattr(FakeBackend, "render", unexpected_render)
    output = tmp_path / "oversized.png"
    with pytest.raises(SystemExit) as error:
        main(["smoke", "--out", str(output), "--width", width, "--height", height])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not output.exists()
