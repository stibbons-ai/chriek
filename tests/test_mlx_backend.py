"""Exercise the MLX backend and its consumers without the optional inference packages."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import running, submit, wait_for
from PIL import Image

from chriek import cli, mlx_backend


class Inference:
    class CacheMiss(Exception):
        pass

    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        self.cache_miss = False
        self.load_error = None
        self.counter_failure = None
        self.probe_timeout = False
        self.lookups = []
        self.loads = []
        self.renders = []
        self.peak = 0

    def download(self, model, **options):
        self.lookups.append((model, options))
        if self.cache_miss and options.get("local_files_only"):
            raise self.CacheMiss
        if not options.get("local_files_only"):
            (self.snapshot / "model.safetensors").touch()
        return str(self.snapshot)

    def load(self, root, *, low_memory):
        self.loads.append((root, low_memory))
        self.peak = max(self.peak, 100)
        if self.load_error is not None:
            raise self.load_error("load failed")
        return self.render

    def reset_peak(self):
        if self.counter_failure == "reset":
            raise RuntimeError("counter unavailable")
        self.peak = 0

    def get_peak(self):
        if self.counter_failure == "read":
            raise RuntimeError("counter unavailable")
        return self.peak

    def probe_gpu(self, *args, **kwargs):
        if self.probe_timeout:
            raise subprocess.TimeoutExpired("system_profiler", 10)
        return SimpleNamespace(stdout='{"SPDisplaysDataType": [{"sppci_cores": "12"}]}')

    def render(self, prompt, references, **options):
        self.renders.append((prompt, references, options))
        self.peak = max(self.peak, 50 + len(references))
        return Image.new("RGBA", (options["width"], options["height"])), {}


@pytest.fixture
def inference(monkeypatch, tmp_path):
    for name in (
        "model.safetensors",
        "ae.safetensors",
        "config.json",
        "text_encoder/config.json",
        "text_encoder/tokenizer.json",
        "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/model-00002-of-00002.safetensors",
    ):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.touch()
    fake = Inference(tmp_path)
    core = SimpleNamespace(
        reset_peak_memory=fake.reset_peak,
        get_peak_memory=fake.get_peak,
        device_info=lambda: {"device_name": "Test chip", "memory_size": 64 * 2**30},
    )
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(
        sys.modules,
        "flux3_mlx",
        SimpleNamespace(Flux3Klein=SimpleNamespace(from_pretrained=fake.load)),
    )
    monkeypatch.setitem(
        sys.modules,
        "flux3_mlx.pipeline",
        SimpleNamespace(DOWNSAMPLING_RATIO=16, MULTI_REFERENCE_MAX_PIXELS=1024**2),
    )
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake.download)
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub.errors",
        SimpleNamespace(LocalEntryNotFoundError=fake.CacheMiss),
    )
    monkeypatch.setattr(mlx_backend, "version", {"flux3-mlx": "0.1.0", "mlx": "0.32.4"}.__getitem__)
    monkeypatch.setattr(
        mlx_backend,
        "subprocess",
        SimpleNamespace(run=fake.probe_gpu, SubprocessError=subprocess.SubprocessError),
    )
    monkeypatch.setattr(mlx_backend.platform, "mac_ver", lambda: ("27.2", (), ""))
    return fake


def test_text_and_edit_pass_exact_inference_parameters_and_share_one_load(inference):
    backend = mlx_backend.MlxBackend()
    assert inference.loads == []
    backend.start_job()
    assert backend.render("a cat", 32, 16, 7).mode == "RGB"
    with Image.new("RGB", (16, 16)) as reference:
        backend.render("a hat", 32, 16, 8, (reference,))
        options = dict(width=32, height=16, steps=16, guidance=None, upsample=False)
        assert inference.renders == [
            ("a cat", [], dict(options, seed=7)),
            ("a hat", [reference], dict(options, seed=8)),
        ]
    assert inference.loads == [(str(inference.snapshot), False)]
    [(model, options)] = inference.lookups
    assert model == "black-forest-labs/flux-3-image-klein"
    assert options["revision"] == "09561231f44e41381c461caf53d5452005a60ac9"
    assert options["local_files_only"] is True
    assert set(options["allow_patterns"]) >= {"model.safetensors", "ae.safetensors"}


def test_edit_pads_small_multi_reference_images(inference, tmp_path):
    small_path = tmp_path / "small.png"
    regular_path = tmp_path / "regular.png"
    for path, size, color in (
        (small_path, (12, 32), "red"),
        (regular_path, (64, 64), "blue"),
    ):
        with Image.new("RGB", size, color) as image:
            image.save(path)

    backend = mlx_backend.MlxBackend()
    with Image.open(small_path) as small, Image.open(regular_path) as regular:
        assert backend.render("edit", 32, 16, 7, (small, regular)).size == (32, 16)
        [(_, (padded, unchanged), _)] = inference.renders
        assert padded.size == (16, 32)
        assert padded.getpixel((8, 16)) == (255, 0, 0)
        assert small.size == (12, 32)
        assert unchanged is regular


def test_edit_keeps_tall_multi_reference_nonempty_after_runtime_crop(inference):
    backend = mlx_backend.MlxBackend()
    with Image.new("RGB", (12, 131072), "red") as tall, Image.new("RGB", (64, 64)) as regular:
        assert backend.render("edit", 32, 16, 7, (tall, regular)).size == (32, 16)
        [(_, references, _)] = inference.renders
        for reference in references:
            width, height = reference.size
            # flux3-mlx 0.1.0 scales to the area cap, then crops to its grid.
            scale = min(1.0, math.sqrt(1024**2 / (width * height)))
            size = tuple(int(side * scale) // 16 * 16 for side in reference.size)
            assert min(size) >= 16, size
        assert references[1] is regular


@pytest.mark.parametrize("load_error", [RuntimeError, SystemExit])
@pytest.mark.parametrize("cache_miss", [True, False], ids=["no-snapshot", "incomplete-snapshot"])
def test_missing_weights_download_and_retry(inference, cache_miss, load_error, monkeypatch):
    monkeypatch.setattr(mlx_backend, "perf_counter", iter([0.0, 1.25, 2.0, 3.0]).__next__)
    backend = mlx_backend.MlxBackend()
    inference.cache_miss = cache_miss
    weights = inference.snapshot / "model.safetensors"
    if not cache_miss:
        weights.unlink()
    inference.load_error = load_error
    backend.start_job()
    with pytest.raises(RuntimeError):
        backend.render("a cat", 32, 16, 7)
    assert [options.get("local_files_only", False) for _, options in inference.lookups] == [
        True,
        False,
    ]
    assert set(inference.lookups[-1][1]["allow_patterns"]) == {
        "model.safetensors",
        "ae.safetensors",
        "config.json",
        "LICENSE.md",
        "NOTICE.md",
        "text_encoder/*",
    }
    assert backend.run_details()["load_seconds"] == 1.25
    assert backend.run_details()["peak_memory_bytes"] == 100

    inference.cache_miss = False
    inference.load_error = None
    backend.start_job()
    backend.render("a cat", 32, 16, 7)
    assert inference.loads == [(str(inference.snapshot), False)] * 2
    assert len(inference.lookups) == 3 and inference.lookups[-1][1]["local_files_only"] is True


def test_measurements_are_per_job_not_per_image_or_process(inference, monkeypatch):
    monkeypatch.setattr(mlx_backend, "perf_counter", iter([2.0, 3.25]).__next__)
    backend = mlx_backend.MlxBackend()
    backend.start_job()
    backend.render("a cat", 32, 16, 7)
    backend.render("a cat", 32, 16, 8)
    assert backend.run_details() == {
        "load_seconds": 1.25,
        "peak_memory_bytes": 100,
        "hardware": {
            "chip": "Test chip",
            "gpu_cores": 12,
            "memory_gb": 64,
            "macos_version": "27.2",
        },
        "runtime": {"flux3_mlx": "0.1.0", "mlx": "0.32.4"},
        "precision": "BF16",
        "steps": 16,
    }

    backend.start_job()
    backend.render("a cat", 32, 16, 7)
    details = backend.run_details()
    assert (details["load_seconds"], details["peak_memory_bytes"]) == (0.0, 50)


@pytest.mark.parametrize("probe_timeout", [False, True], ids=["gpu-cores", "probe-timeout"])
def test_smoke_prints_mlx_job_metrics(
    inference, tmp_path, monkeypatch, capsys, caplog, probe_timeout
):
    for name in os.environ:
        if name.upper().startswith("CHRIEK_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("CHRIEK_BACKEND", "mlx")
    monkeypatch.setattr(mlx_backend, "perf_counter", iter([2.0, 3.25]).__next__)
    inference.peak = 200  # Prior process activity must not count toward this job's peak.
    inference.probe_timeout = probe_timeout
    output = tmp_path / "smoke.png"

    assert cli.main(["smoke", "--out", str(output), "--width", "16", "--height", "16"]) == 0
    with Image.open(output) as image:
        assert (image.format, image.size) == ("PNG", (16, 16))
    assert ("system_profiler" in caplog.text and "timed out" in caplog.text) == probe_timeout

    record = json.loads(capsys.readouterr().out)
    assert record["load_seconds"] == 1.25
    assert record["peak_memory_bytes"] == 100
    assert record["hardware"] == {
        "chip": "Test chip",
        "gpu_cores": None if probe_timeout else 12,
        "memory_gb": 64,
        "macos_version": "27.2",
    }
    assert record["runtime"] == {"flux3_mlx": "0.1.0", "mlx": "0.32.4"}
    assert record["precision"] == "BF16"
    assert record["steps"] == 16


@pytest.mark.parametrize("counter_failure", ["reset", "read"])
@pytest.mark.local_storage
def test_peak_counter_failure_does_not_discard_images(inference, tmp_path, caplog, counter_failure):
    backend = mlx_backend.MlxBackend()
    with running(tmp_path, backend=backend) as client:
        inference.counter_failure = counter_failure
        job = wait_for(client, submit(client, width=16, height=16)["id"])
        assert job["status"] == "succeeded"
        assert job["error"] is None
        assert [image["index"] for image in job["images"]] == [0]
        assert client.get(job["images"][0]["url"]).content.startswith(b"\x89PNG")
        run = job["run"]
        assert run["peak_memory_bytes"] is None
        assert run["load_seconds"] >= 0.0
        assert run["hardware"]["chip"] == "Test chip"
        assert run["runtime"] == {"flux3_mlx": "0.1.0", "mlx": "0.32.4"}
        assert (run["precision"], run["steps"]) == ("BF16", 16)
        assert "counter unavailable" in caplog.text

        inference.counter_failure = None
        recovered = wait_for(client, submit(client, width=16, height=16)["id"])
        assert recovered["run"]["peak_memory_bytes"] == 50
