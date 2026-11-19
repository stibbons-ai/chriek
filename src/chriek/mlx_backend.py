"""Pinned BF16 FLUX 3 Image Klein inference on Apple Silicon."""

import json
import logging
import math
import platform
import subprocess
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageOps

from chriek.config import ConfigError

log = logging.getLogger(__name__)

# HF caches created before tree listings were recorded may return a partial snapshot.
_MODEL_FILES = (
    "model.safetensors",
    "ae.safetensors",
    "config.json",
    "text_encoder/config.json",
    "text_encoder/tokenizer.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
)
_ALLOW_PATTERNS = (
    "model.safetensors",
    "ae.safetensors",
    "config.json",
    "LICENSE.md",
    "NOTICE.md",
    "text_encoder/*",
)


def _complete(root: Path) -> bool:
    return all((root / name).is_file() for name in _MODEL_FILES)


def _hardware() -> dict[str, object]:
    import mlx.core as mx

    device = mx.device_info()
    try:
        display = subprocess.run(  # noqa: S603 -- fixed macOS executable and arguments
            ["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        gpu_cores = next(
            (
                int(gpu["sppci_cores"])
                for gpu in json.loads(display.stdout)["SPDisplaysDataType"]
                if "sppci_cores" in gpu
            ),
            None,
        )
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        KeyError,
        TypeError,
        OverflowError,
    ) as error:
        log.warning("could not read GPU core count from system_profiler: %s", error)
        gpu_cores = None
    return {
        "chip": device["device_name"],
        "gpu_cores": gpu_cores,
        "memory_gb": round(device["memory_size"] / 2**30),
        "macos_version": platform.mac_ver()[0],
    }


def _fit_multi_reference(image: Image.Image) -> Image.Image:
    from flux3_mlx.pipeline import DOWNSAMPLING_RATIO, MULTI_REFERENCE_MAX_PIXELS

    grid, budget = DOWNSAMPLING_RATIO, MULTI_REFERENCE_MAX_PIXELS
    short, long = sorted(image.size)
    area = short * long
    # Leave references alone if the runtime's area scaling still leaves a full grid cell.
    if short >= grid and (area <= budget or short * math.sqrt(budget / area) >= grid):
        return image
    image = ImageOps.exif_transpose(image).convert("RGB")
    max_side = budget // grid
    if long > max_side:
        size = tuple(max(1, side * max_side // long) for side in image.size)
        image = image.resize(size, Image.Resampling.LANCZOS)
    dx, dy = max(grid - image.width, 0), max(grid - image.height, 0)
    return ImageOps.expand(image, (dx // 2, dy // 2, dx - dx // 2, dy - dy // 2))


class MlxBackend:
    name = "mlx"
    model = "black-forest-labs/flux-3-image-klein"
    revision = "09561231f44e41381c461caf53d5452005a60ac9"
    steps = 16

    def __init__(self):
        try:
            import mlx.core as mx
            from flux3_mlx import Flux3Klein
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError
        except (ImportError, OSError) as error:
            raise ConfigError(
                "CHRIEK_BACKEND=mlx requires the MLX extra on Apple Silicon: "
                "uv sync --locked --extra mlx"
            ) from error

        self._mx = mx
        self._load = Flux3Klein.from_pretrained
        self._download = snapshot_download
        self._cache_miss = LocalEntryNotFoundError
        try:
            self._hardware = _hardware()
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            raise ConfigError("cannot read MLX hardware information") from error
        self._runtime = {"flux3_mlx": version("flux3-mlx"), "mlx": version("mlx")}
        self._pipeline = None
        self._load_seconds = 0.0
        self._peak_reset = False

    def _snapshot(self) -> str:
        try:
            root = self._download(
                self.model,
                revision=self.revision,
                allow_patterns=_MODEL_FILES,
                local_files_only=True,
            )
            if _complete(Path(root)):
                return root
        except self._cache_miss:
            pass
        root = self._download(self.model, revision=self.revision, allow_patterns=_ALLOW_PATTERNS)
        if not _complete(Path(root)):
            raise RuntimeError("FLUX weights are incomplete; check HF_HOME and HF_HUB_OFFLINE")
        return root

    def start_job(self) -> None:
        self._load_seconds = 0.0
        self._peak_reset = False
        try:
            self._mx.reset_peak_memory()
        except Exception:
            log.exception("could not reset the MLX peak memory counter")
        else:
            self._peak_reset = True

    def run_details(self) -> dict[str, object]:
        peak = None
        if self._peak_reset:
            try:
                peak = self._mx.get_peak_memory()
            except Exception:
                log.exception("could not read the MLX peak memory counter")
        return {
            "load_seconds": self._load_seconds,
            "peak_memory_bytes": peak,
            "hardware": self._hardware,
            "runtime": self._runtime,
            "precision": "BF16",
            "steps": self.steps,
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
                self._pipeline = self._load(self._snapshot(), low_memory=False)
            except SystemExit as error:
                # flux3-mlx uses SystemExit for model format errors; don't terminate the worker.
                raise RuntimeError("flux3-mlx could not load the model") from error
            finally:
                self._load_seconds = round(perf_counter() - started, 3)
        image, _ = self._pipeline(
            prompt,
            [_fit_multi_reference(ref) for ref in references]
            if len(references) > 1
            else list(references),
            width=width,
            height=height,
            steps=self.steps,
            guidance=None,
            seed=seed,
            upsample=False,
        )
        return image if image.mode == "RGB" else image.convert("RGB")
