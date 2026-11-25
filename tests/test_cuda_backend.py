"""Exercise the CUDA backend and its consumers without torch, diffusers or a GPU."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from chriek import cli, cuda_backend
from chriek.backend import create_backend
from chriek.config import BackendSettings, ConfigError

REVISION = "09561231f44e41381c461caf53d5452005a60ac9"
BF16 = object()


class Inference:
    """Records what the backend asks of huggingface_hub, diffusers and torch."""

    class CacheMiss(Exception):
        pass

    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        self.cache_miss = False
        self.unloadable = set()
        self.lookups = []
        self.popped = []
        self.pipelines = []
        self.offloads = []
        self.calls = []
        self.reference_pixels = []
        self.peak = 0

    def download(self, model, **options):
        self.lookups.append((model, options))
        if self.cache_miss:
            raise self.CacheMiss
        return str(self.snapshot)

    def blocks(self):
        inference = self

        class Blocks:
            sub_blocks = {"prompt_upsampler": "pup", "text_encoder": "te"}

            def init_pipeline(self, root, components_manager=None):
                inference.popped = [n for n in ("prompt_upsampler",) if n not in self.sub_blocks]
                pipeline = Pipeline(inference, root, components_manager)
                inference.pipelines.append(pipeline)
                return pipeline

        return Blocks()

    def manager(self):
        inference = self

        class Manager:
            def enable_auto_cpu_offload(self, **options):
                inference.offloads.append(options)

        return Manager()


class Pipeline:
    def __init__(self, inference, root, manager):
        self.inference = inference
        self.root = root
        self.manager = manager
        self.loads = []
        self.devices = []

    def load_components(self, **options):
        self.loads.append(options)
        for name in ("text_encoder", "tokenizer", "transformer", "vae"):
            if name not in self.inference.unloadable:
                setattr(self, name, name)

    def to(self, device):
        self.devices.append(device)

    def __call__(self, **options):
        self.inference.calls.append(options)
        # Read while borrowed: the smoke command closes references after rendering.
        self.inference.reference_pixels.append(
            [image.getpixel((0, 0)) for image in options.get("image", ())]
        )
        self.inference.peak = 100 + len(options.get("image", ()))
        return [Image.new("RGBA", (options["width"], options["height"]))]


@pytest.fixture
def inference(monkeypatch, tmp_path):
    fake = Inference(tmp_path)

    class Generator:
        def manual_seed(self, seed):
            self.seed = seed
            return self

    def reset_peak():
        fake.peak = 0

    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda index: "Test GPU",
        get_device_properties=lambda index: SimpleNamespace(total_memory=24 * 2**30),
        reset_peak_memory_stats=reset_peak,
        max_memory_allocated=lambda: fake.peak,
    )
    torch = SimpleNamespace(
        bfloat16=BF16, Generator=Generator, cuda=cuda, version=SimpleNamespace(cuda="13.0")
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(ComponentsManager=fake.manager, Flux3DistilledAutoBlocks=fake.blocks),
    )
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake.download)
    )
    errors = SimpleNamespace(LocalEntryNotFoundError=fake.CacheMiss)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    versions = {"torch": "2.15.0", "diffusers": "0.43.0", "transformers": "5.24.1"}
    versions["accelerate"] = "1.16.0"
    monkeypatch.setattr(cuda_backend, "version", versions.__getitem__)
    return fake


@pytest.mark.parametrize("offload", ["none", "sequential"])
def test_text_and_ordered_edit_share_one_cached_bf16_pipeline(inference, offload):
    backend = create_backend(BackendSettings("cuda", offload))
    assert inference.lookups == []

    assert backend.render("a cat", 32, 16, 7).mode == "RGB"
    colors = ("red", "green", "blue")
    references = [Image.new("RGB", (64, 48), color) for color in colors]
    assert backend.render("a hat", 32, 16, 8, references).size == (32, 16)
    backend.render("a cat", 32, 16, 9)

    text, edit, repeat = inference.calls
    common = dict(width=32, height=16, num_inference_steps=16, output="images")
    assert {k: v for k, v in text.items() if k != "generator"} == dict(common, prompt="a cat")
    assert (text["generator"].seed, edit["generator"].seed, repeat["generator"].seed) == (7, 8, 9)
    assert edit["image"] == references and edit["image"] is not references
    assert all(a is b for a, b in zip(edit["image"], references, strict=True))
    for reference, color in zip(references, colors, strict=True):
        assert reference.getpixel((0, 0)) == Image.new("RGB", (1, 1), color).getpixel((0, 0))
        assert reference.size == (64, 48)

    [(model, options)] = inference.lookups
    assert (model, options) == (
        "black-forest-labs/flux-3-image-klein",
        {"revision": REVISION, "local_files_only": True},
    )
    assert inference.popped == ["prompt_upsampler"]
    [pipeline] = inference.pipelines
    assert pipeline.root == str(inference.snapshot)
    assert pipeline.loads == [{"dtype": BF16, "local_files_only": True}]
    if offload == "none":
        assert (pipeline.manager, pipeline.devices, inference.offloads) == (None, ["cuda"], [])
    else:
        assert pipeline.devices == []
        [options] = inference.offloads
        strategy = options.pop("offload_strategy")
        assert options == {"device": "cuda"}
        encoder, vae = object(), object()
        resident = [encoder, vae]
        hooks = strategy(hooks=resident, model_id="transformer", model=None, execution_device=0)
        assert hooks == [encoder, vae]


def test_missing_or_incomplete_weights_fail_without_downloading(inference):
    backend = create_backend(BackendSettings("cuda", "sequential"))
    inference.cache_miss = True
    with pytest.raises(RuntimeError, match="not in the Hugging Face cache"):
        backend.render("a cat", 32, 16, 7)
    assert inference.pipelines == []

    inference.cache_miss = False
    inference.unloadable = {"transformer"}
    with pytest.raises(RuntimeError, match="could not load transformer"):
        backend.render("a cat", 32, 16, 7)
    assert inference.calls == [] and inference.offloads == []

    inference.unloadable = set()
    backend.render("a cat", 32, 16, 7)
    assert len(inference.pipelines) == 2
    assert all(options["local_files_only"] for _, options in inference.lookups)


@pytest.fixture
def smoke_env(monkeypatch):
    for name in os.environ:
        if name.upper().startswith("CHRIEK_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("CHRIEK_BACKEND", "cuda")
    monkeypatch.setenv("CHRIEK_OFFLOAD", "sequential")


def test_smoke_three_reference_edit_prints_cuda_run_record(
    inference, smoke_env, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(cuda_backend, "perf_counter", iter([2.0, 3.25]).__next__)
    paths = []
    for index, color in enumerate(("red", "green", "blue")):
        paths += ["--reference", str(tmp_path / f"{index}.png")]
        Image.new("RGB", (64, 64), color).save(paths[-1])
    output = tmp_path / "edit.png"

    args = ["smoke", "--out", str(output), "--width", "32", "--height", "16", *paths]
    assert cli.main(args) == 0

    with Image.open(output) as image:
        assert (image.format, image.mode, image.size) == ("PNG", "RGB", (32, 16))
        assert image.info["Model-Revision"] == REVISION
    assert inference.reference_pixels == [[(255, 0, 0), (0, 128, 0), (0, 0, 255)]]
    record = json.loads(capsys.readouterr().out)
    assert record.pop("render_seconds") >= 0
    assert record == {
        "backend": "cuda",
        "model": "black-forest-labs/flux-3-image-klein",
        "revision": REVISION,
        "load_seconds": 1.25,
        "peak_memory_bytes": 103,
        "hardware": {"gpu": "Test GPU", "memory_gb": 24, "cuda": "13.0"},
        "runtime": {
            "torch": "2.15.0",
            "diffusers": "0.43.0",
            "transformers": "5.24.1",
            "accelerate": "1.16.0",
        },
        "precision": "BF16",
        "steps": 16,
        "offload": "sequential",
        "prompt": "A red bicycle in the rain",
        "seed": 7,
        "width": 32,
        "height": 16,
        "references": 3,
        "output": str(output),
    }


@pytest.mark.parametrize(
    ("module", "message"),
    [("torch", "uv sync --locked --extra cuda"), ("gpu", "NVIDIA GPU visible to PyTorch")],
)
def test_smoke_without_cuda_runtime_is_a_configuration_error(
    inference, smoke_env, tmp_path, monkeypatch, capsys, module, message
):
    if module == "torch":
        monkeypatch.setitem(sys.modules, "torch", None)
    else:
        monkeypatch.setattr(sys.modules["torch"].cuda, "is_available", lambda: False)
    output = tmp_path / "out.png"
    assert cli.main(["smoke", "--out", str(output), "--width", "16", "--height", "16"]) == 2
    assert message in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("backend", ["fake", "mlx"])
def test_sequential_offload_needs_cuda(backend):
    with pytest.raises(ConfigError, match="'sequential' with CHRIEK_BACKEND=cuda"):
        BackendSettings.from_env({"CHRIEK_BACKEND": backend, "CHRIEK_OFFLOAD": "sequential"})
    assert BackendSettings.from_env({"CHRIEK_BACKEND": "cuda", "CHRIEK_OFFLOAD": "sequential"}) == (
        BackendSettings("cuda", "sequential")
    )
