"""Exercise the CUDA backend and its consumers without torch, diffusers or a GPU."""

import json
import math
import os
import subprocess
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
COMPONENTS = ("text_encoder", "tokenizer", "transformer", "vae")


class Inference:
    """Records what the backend asks of huggingface_hub, diffusers and torch."""

    class CacheMiss(Exception):
        pass

    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        self.cache_miss = False
        self.unloadable = set()
        # Component name -> (pretrained_model_name_or_path, subfolder) diffusers resolved.
        self.sources = {}
        self.lookups = []
        self.popped = []
        self.pipelines = []
        self.offloads = []
        self.events = []
        self.calls = []
        self.reference_pixels = []
        self.peak = 0

    def download(self, model, **options):
        self.lookups.append((model, options))
        if self.cache_miss:
            raise self.CacheMiss("IncompleteSnapshotError: 3 file(s) are missing")
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
            def __init__(self):
                self.components = {}

            def add(self, name, component):
                self.components[name] = component

            def enable_auto_cpu_offload(self, **options):
                inference.events.append("offload")
                inference.offloads.append((self, options))

        return Manager()


class Pipeline:
    def __init__(self, inference, root, manager):
        self.inference = inference
        self.root = root
        self.manager = manager
        self.loads = []

    def get_component_spec(self, name):
        # The real snapshot keeps the tokenizer's files in text_encoder/.
        default = (self.root, "text_encoder" if name == "tokenizer" else name)
        path, subfolder = self.inference.sources.get(name, default)
        return SimpleNamespace(pretrained_model_name_or_path=path, subfolder=subfolder)

    def load_components(self, **options):
        self.inference.events.append("load")
        self.loads.append(options)
        for name in COMPONENTS:
            if name not in self.inference.unloadable:
                setattr(self, name, name)
                # Like diffusers: only a manager given to init_pipeline receives the components.
                if self.manager is not None:
                    self.manager.add(name, name)

    def to(self, device):
        self.inference.events.append(f"to:{device}")

    def __call__(self, **options):
        self.inference.calls.append(options)
        # Read while borrowed: the smoke command closes references after rendering.
        self.inference.reference_pixels.append(
            [image.getpixel((0, 0)) for image in options.get("image", ())]
        )
        # Like torch's allocator peak: only reset_peak_memory_stats lowers it.
        self.inference.peak = max(self.inference.peak, 100 + len(options.get("image", ())))
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
        {
            "revision": REVISION,
            # The README's download: the diffusers layout only, not the whole repo.
            "allow_patterns": (
                "modular_model_index.json",
                "transformer/*",
                "vae/*",
                "text_encoder/*",
                "LICENSE.md",
                "NOTICE.md",
            ),
            "local_files_only": True,
        },
    )
    assert inference.popped == ["prompt_upsampler"]
    [pipeline] = inference.pipelines
    assert pipeline.root == str(inference.snapshot)
    assert pipeline.loads == [{"dtype": BF16, "local_files_only": True}]
    # Offload hooks only components registered when it is enabled, so loading comes first.
    if offload == "none":
        assert pipeline.manager is None
        assert inference.events == ["load", "to:cuda"]
    else:
        assert inference.events == ["load", "offload"]
        [(manager, options)] = inference.offloads
        # Offload hooks only the components of the manager the pipeline registered them with.
        assert manager is pipeline.manager
        assert sorted(manager.components) == sorted(COMPONENTS)
        strategy = options.pop("offload_strategy")
        assert options == {"device": "cuda"}
        # diffusers passes the hooks still on the GPU as each component arrives, in the pipeline's
        # real order, and offloads exactly the hooks the strategy returns.
        hooks = {
            name: SimpleNamespace(model_id=name) for name in ("text_encoder", "vae", "transformer")
        }
        resident = []
        for arriving in ("text_encoder", "vae", "transformer", "vae"):
            others = [hook for hook in resident if hook.model_id != arriving]
            kwargs = {"model_id": arriving, "model": None, "execution_device": 0}
            assert strategy(hooks=others, **kwargs) == others, arriving
            resident = [hooks[arriving]]
        # Even if the encoder were still resident when the transformer arrives, it leaves first.
        both = [hooks["text_encoder"], hooks["vae"]]
        kwargs = {"model_id": "transformer", "model": None, "execution_device": 0}
        assert strategy(hooks=both, **kwargs) == both


def test_missing_weights_fail_without_downloading(inference):
    backend = create_backend(BackendSettings("cuda", "sequential"))
    inference.cache_miss = True
    with pytest.raises(RuntimeError, match="not in the Hugging Face cache.*3 file.s. are missing"):
        backend.render("a cat", 32, 16, 7)
    assert inference.pipelines == []
    assert all(options["local_files_only"] for _, options in inference.lookups)


@pytest.mark.parametrize("name", COMPONENTS)
def test_unloadable_component_fails_before_rendering(inference, name):
    backend = create_backend(BackendSettings("cuda", "sequential"))
    inference.unloadable = {name}
    with pytest.raises(RuntimeError, match=f"could not load {name}"):
        backend.render("a cat", 32, 16, 7)
    assert inference.calls == [] and inference.events == ["load"]

    inference.unloadable = set()
    backend.render("a cat", 32, 16, 7)
    assert len(inference.pipelines) == 2


@pytest.mark.parametrize(
    "source",
    [
        ("black-forest-labs/flux-3-image-klein", "text_encoder"),
        (None, "text_encoder"),
        ("SNAPSHOT", "../other-snapshot/text_encoder"),
    ],
    ids=["hub-repo", "no-source", "outside-snapshot"],
)
@pytest.mark.parametrize("name", COMPONENTS)
def test_component_resolved_outside_the_pinned_snapshot_is_never_loaded(inference, source, name):
    path, subfolder = source
    path = str(inference.snapshot) if path == "SNAPSHOT" else path
    # The real snapshot keeps the tokenizer's files in text_encoder/.
    folder = "text_encoder" if name == "tokenizer" else name
    inference.sources = {name: (path, subfolder.replace("text_encoder", folder))}
    backend = create_backend(BackendSettings("cuda", "none"))
    with pytest.raises(RuntimeError, match=f"^{name} not found in the pinned snapshot"):
        backend.render("a cat", 32, 16, 7)
    assert inference.events == [] and inference.calls == []


def test_edit_fits_tiny_and_tall_references_without_touching_them(inference):
    backend = create_backend(BackendSettings("cuda", "none"))
    with (
        Image.new("RGB", (4, 4), "red") as tiny,
        Image.new("RGB", (12, 131072), "red") as tall,
        Image.new("RGB", (64, 48), "blue") as regular,
    ):
        backend.render("edit", 32, 16, 7, (tiny, tall, regular))
        [fitted] = [call["image"] for call in inference.calls]
        for image in fitted:
            # diffusers scales several references under 1 MP, then crops them to its 16 px grid.
            scale = min(1.0, math.sqrt(1024**2 / (image.width * image.height)))
            assert min(int(side * scale) // 16 * 16 for side in image.size) >= 16, image.size
        assert fitted[0] is not tiny and fitted[1] is not tall and fitted[2] is regular
        assert fitted[0].getpixel((8, 8)) == (255, 0, 0)
        assert (tiny.size, tall.size) == ((4, 4), (12, 131072))
        for image, color in ((tiny, (255, 0, 0)), (tall, (255, 0, 0)), (regular, (0, 0, 255))):
            assert image.getextrema() == tuple((value, value) for value in color)


def test_single_reference_is_passed_as_is(inference):
    # diffusers fits one reference itself (fill, up to 4 MP); padding it here would add black.
    backend = create_backend(BackendSettings("cuda", "none"))
    with Image.new("RGBA", (4, 4), (255, 0, 0, 128)) as tiny:
        backend.render("edit", 32, 16, 7, [tiny])
        [call] = inference.calls
        assert call["image"] == [tiny] and call["image"][0] is tiny
        assert (tiny.size, tiny.mode) == ((4, 4), "RGBA")
        assert tiny.tobytes() == Image.new("RGBA", (4, 4), (255, 0, 0, 128)).tobytes()


def test_edit_upright_rgb_fits_small_references_and_borrows_large_ones(inference, tmp_path):
    # Left half red, right half blue; EXIF orientation 6 displays it rotated 90° clockwise.
    sideways = Image.new("RGBA", (8, 4), (255, 0, 0, 128))
    sideways.paste((0, 0, 255, 128), (4, 0, 8, 4))
    exif = Image.Exif()
    exif[0x0112] = 6
    sideways.save(tmp_path / "sideways.png", exif=exif)
    backend = create_backend(BackendSettings("cuda", "none"))
    with (
        Image.open(tmp_path / "sideways.png") as small,
        Image.new("RGB", (2048, 2048), "green") as large,
    ):
        backend.render("edit", 32, 16, 7, (small, large))
        [(fitted, unchanged)] = [call["image"] for call in inference.calls]
        assert unchanged is large
        # Upright 4×8 (red over blue), padded to 16×16 at offset (6, 4).
        assert (fitted.size, fitted.mode) == ((16, 16), "RGB")
        assert fitted.getpixel((7, 5)) == (255, 0, 0)
        assert fitted.getpixel((7, 10)) == (0, 0, 255)
        assert fitted.getpixel((0, 0)) == (0, 0, 0)
        assert (small.size, small.mode) == ((8, 4), "RGBA")
        assert small.tobytes() == sideways.tobytes()


@pytest.mark.parametrize("offload", ["Sequential", "model", ""])
def test_cuda_rejects_unknown_offload(offload):
    with pytest.raises(ConfigError, match="CHRIEK_OFFLOAD must be"):
        BackendSettings.from_env({"CHRIEK_BACKEND": "cuda", "CHRIEK_OFFLOAD": offload})
    assert BackendSettings.from_env({"CHRIEK_BACKEND": "cuda", "CHRIEK_OFFLOAD": "none"}) == (
        BackendSettings("cuda", "none")
    )


def test_measurements_are_per_job_not_per_image_or_process(inference, monkeypatch):
    monkeypatch.setattr(cuda_backend, "perf_counter", iter([2.0, 3.25]).__next__)
    backend = create_backend(BackendSettings("cuda", "none"))
    backend.start_job()
    references = [Image.new("RGB", (64, 48)) for _ in range(3)]
    backend.render("a hat", 32, 16, 7, references)
    backend.render("a cat", 32, 16, 8)
    details = backend.run_details()
    assert (details["load_seconds"], details["peak_memory_bytes"]) == (1.25, 103)

    backend.start_job()
    backend.render("a cat", 32, 16, 7)
    details = backend.run_details()
    assert (details["load_seconds"], details["peak_memory_bytes"]) == (0.0, 100)


def test_fake_and_mlx_never_import_cuda_libraries():
    # Records import attempts, so it holds whether or not the cuda extra is installed.
    code = """
import sys
attempted = []
CUDA = {"torch", "torchvision", "diffusers", "transformers", "accelerate"}
class Spy:
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in CUDA:
            attempted.append(name)
sys.meta_path.insert(0, Spy())
from chriek import cli
from chriek.backend import create_backend
from chriek.config import BackendSettings, ConfigError
for name in ("fake", "mlx"):
    try:
        create_backend(BackendSettings(name))
    except ConfigError:
        pass
assert not attempted, attempted
"""
    subprocess.run([sys.executable, "-c", code], check=True)


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
    [("torch", "uv sync --locked --extra cuda"), ("gpu", "driver 580 or newer")],
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
