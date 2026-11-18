"""Exercise the image backend against stand-ins for the optional inference packages."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from chriek import mlx_backend


class Inference:
    class CacheMiss(Exception):
        pass

    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        self.cache_miss = False
        self.fail_load = False
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
        self.peak = 100
        if self.fail_load:
            raise RuntimeError("load failed")
        return self.render

    def reset_peak(self):
        self.peak = 0

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
    core = SimpleNamespace(reset_peak_memory=fake.reset_peak, get_peak_memory=lambda: fake.peak)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(
        sys.modules,
        "flux3_mlx",
        SimpleNamespace(Flux3Klein=SimpleNamespace(from_pretrained=fake.load)),
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
        "_hardware",
        lambda: {
            "chip": "Test chip",
            "gpu_cores": 12,
            "memory_gb": 64,
            "macos_version": "27.2",
        },
    )
    return fake


def test_text_and_edit_pass_exact_inference_parameters_and_share_one_load(inference):
    backend = mlx_backend.MlxBackend(low_memory=False)
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


@pytest.mark.parametrize("cache_miss", [True, False], ids=["no-snapshot", "incomplete-snapshot"])
def test_missing_weights_download_and_retry(inference, cache_miss, monkeypatch):
    monkeypatch.setattr(mlx_backend, "perf_counter", iter([0.0, 1.25, 2.0, 3.0]).__next__)
    backend = mlx_backend.MlxBackend(low_memory=True)
    inference.cache_miss = cache_miss
    weights = inference.snapshot / "model.safetensors"
    if not cache_miss:
        weights.unlink()
    inference.fail_load = True
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

    inference.cache_miss = inference.fail_load = False
    backend.start_job()
    backend.render("a cat", 32, 16, 7)
    assert inference.loads == [(str(inference.snapshot), True)] * 2
    assert len(inference.lookups) == 3 and inference.lookups[-1][1]["local_files_only"] is True


def test_measurements_are_per_job_not_per_image_or_process(inference, monkeypatch):
    monkeypatch.setattr(mlx_backend, "perf_counter", iter([2.0, 3.25]).__next__)
    backend = mlx_backend.MlxBackend(low_memory=False)
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
