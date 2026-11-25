"""Pinned BF16 FLUX 3 Image Klein inference on NVIDIA GPUs through diffusers' modular pipeline."""

from collections.abc import Sequence
from importlib.metadata import version
from time import perf_counter

from PIL import Image

from chriek.backend import FLUX_MODEL, FLUX_REVISION, FLUX_STEPS, fit_references
from chriek.config import ConfigError

# The components the pipeline needs once the prompt upsampler step is removed; load_components
# only logs a warning for a component it cannot load.
_COMPONENTS = ("text_encoder", "tokenizer", "transformer", "vae")
# diffusers' Flux3ImageProcessor fits each of several references under 1 MP, then crops to 16 px.
_GRID, _BUDGET = 16, 1024 * 1024


def _offload_all_resident(hooks, **_):
    """Offload strategy for `sequential`: before a component runs, every other one leaves the GPU,
    so the text encoder is on the CPU before the transformer is moved in."""
    return hooks


class CudaBackend:
    name = "cuda"
    model = FLUX_MODEL
    revision = FLUX_REVISION
    steps = FLUX_STEPS

    def __init__(self, offload: str):
        try:
            import torch
            from diffusers import ComponentsManager, Flux3DistilledAutoBlocks
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError
        except (ImportError, OSError) as error:
            raise ConfigError(
                "CHRIEK_BACKEND=cuda requires the CUDA extra on Linux: "
                "uv sync --locked --extra cuda"
            ) from error
        if not torch.cuda.is_available():
            raise ConfigError("CHRIEK_BACKEND=cuda requires an NVIDIA GPU visible to PyTorch")

        self._torch = torch
        self._blocks = Flux3DistilledAutoBlocks
        self._manager = ComponentsManager
        self._download = snapshot_download
        self._cache_miss = LocalEntryNotFoundError
        self._offload = offload
        self._hardware = {
            "gpu": torch.cuda.get_device_name(0),
            "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30),
            "cuda": torch.version.cuda,
        }
        self._runtime = {
            name: version(name) for name in ("torch", "diffusers", "transformers", "accelerate")
        }
        self._pipeline = None
        self._load_seconds = 0.0

    def _load(self):
        try:
            # Cache only: the operator downloads the gated weights with their own token beforehand.
            root = self._download(self.model, revision=self.revision, local_files_only=True)
        except self._cache_miss as error:
            raise RuntimeError(
                f"{self.model} at revision {self.revision} is not in the Hugging Face cache; "
                "download it first (see README.md)"
            ) from error
        blocks = self._blocks()
        # chriek sends the prompt as given, as the MLX backend does; no second model is loaded.
        blocks.sub_blocks.pop("prompt_upsampler")
        manager = self._manager() if self._offload == "sequential" else None
        pipeline = blocks.init_pipeline(root, components_manager=manager)
        pipeline.load_components(dtype=self._torch.bfloat16, local_files_only=True)
        missing = [name for name in _COMPONENTS if getattr(pipeline, name, None) is None]
        if missing:
            raise RuntimeError(f"could not load {', '.join(missing)} from {root}")
        if manager is None:
            pipeline.to("cuda")
        else:
            manager.enable_auto_cpu_offload(device="cuda", offload_strategy=_offload_all_resident)
        return pipeline

    def start_job(self) -> None:
        self._load_seconds = 0.0
        self._torch.cuda.reset_peak_memory_stats()

    def run_details(self) -> dict[str, object]:
        return {
            "load_seconds": self._load_seconds,
            "peak_memory_bytes": self._torch.cuda.max_memory_allocated(),
            "hardware": self._hardware,
            "runtime": self._runtime,
            "precision": "BF16",
            "steps": self.steps,
            "offload": self._offload,
        }

    def render(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        references: Sequence[Image.Image] = (),
    ) -> Image.Image:
        if self._pipeline is None:
            started = perf_counter()
            try:
                self._pipeline = self._load()
            finally:
                self._load_seconds = round(perf_counter() - started, 3)
        options = {"image": fit_references(references, _GRID, _BUDGET)} if references else {}
        [image] = self._pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=self.steps,
            generator=self._torch.Generator().manual_seed(seed),
            output="images",
            **options,
        )
        return image if image.mode == "RGB" else image.convert("RGB")
