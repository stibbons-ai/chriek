"""Image backend seam and configured backend selection."""

import hashlib
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from PIL import Image, ImageDraw

from chriek.config import BackendSettings


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


def create_backend(settings: BackendSettings) -> ImageBackend:
    if settings.name == "mlx":
        from chriek.mlx_backend import MlxBackend

        return MlxBackend(low_memory=settings.offload == "sequential")
    return FakeBackend()
