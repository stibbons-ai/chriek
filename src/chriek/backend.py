"""Image backend seam and configured backend selection."""

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from PIL import Image, ImageDraw, ImageOps

from chriek.config import BackendSettings

# Model identity shared by the real backends: the same pinned BF16 checkpoint on every platform.
FLUX_MODEL = "black-forest-labs/flux-3-image-klein"
FLUX_REVISION = "09561231f44e41381c461caf53d5452005a60ac9"
FLUX_STEPS = 16


class ImageBackend(Protocol):
    name: str
    model: str
    revision: str

    def render(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        references: Sequence[Image.Image] = (),
    ) -> Image.Image:
        """Render one image without mutating or closing the borrowed references."""
        ...


@runtime_checkable
class JobMetrics(Protocol):
    def start_job(self) -> None: ...

    def run_details(self) -> dict[str, object]: ...


def run_record(
    backend: ImageBackend, elapsed: float, details: dict[str, object]
) -> dict[str, object]:
    return {
        "backend": backend.name,
        "model": backend.model,
        "revision": backend.revision,
        "render_seconds": round(elapsed, 3),
        **details,
    }


class FakeBackend:
    """Deterministic placeholder: prompt, size, seed and references determine the pixels."""

    name = "fake"
    model = "chriek-fake-backend"
    revision = "fake-v1"

    def render(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        references: Sequence[Image.Image] = (),
    ) -> Image.Image:
        hasher = hashlib.sha256(f"{prompt}\n{seed}\n{width}x{height}".encode())
        if references:
            hasher.update(f"\n{len(references)}".encode())
        for index, reference in enumerate(references):
            rgb = reference if reference.mode == "RGB" else reference.convert("RGB")
            hasher.update(f"\nref:{index}:{rgb.width}x{rgb.height}:".encode())
            hasher.update(rgb.tobytes())
        digest = hasher.digest()
        image = Image.new("RGB", (width, height), tuple(digest[0:3]))
        inset_x = width * (1 + digest[6] % 3) // 8
        inset_y = height * (1 + digest[7] % 3) // 8
        ImageDraw.Draw(image).rectangle(
            (inset_x, inset_y, width - inset_x - 1, height - inset_y - 1), fill=tuple(digest[3:6])
        )
        return image


def fit_references(references: Sequence[Image.Image], grid: int, budget: int) -> list[Image.Image]:
    """A new list of the borrowed references, in order. One reference is passed as is; each of
    several is padded or shrunk so the runtime's area scaling and grid crop (to `budget` pixels,
    then down to multiples of `grid`) leave it at least one grid cell."""
    if len(references) < 2:
        return list(references)
    return [_fit_multi_reference(image, grid, budget) for image in references]


def _fit_multi_reference(image: Image.Image, grid: int, budget: int) -> Image.Image:
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


def create_backend(settings: BackendSettings) -> ImageBackend:
    if settings.name == "mlx":
        from chriek.mlx_backend import MlxBackend

        return MlxBackend()
    if settings.name == "cuda":
        from chriek.cuda_backend import CudaBackend

        return CudaBackend(settings.offload)
    return FakeBackend()
