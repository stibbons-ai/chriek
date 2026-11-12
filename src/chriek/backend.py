"""Image backend seam. Only the deterministic fake backend exists so far; it is not a model."""

import hashlib
from collections.abc import Sequence
from typing import Protocol

from PIL import Image, ImageDraw


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
        """Render one image. `references` are borrowed, read-only images shared by every
        render of a job: implementations must not mutate or close them (copy first to edit)."""
        ...


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
