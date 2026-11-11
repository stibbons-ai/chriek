"""Image backend seam. Only the deterministic fake backend exists so far; it is not a model."""

import hashlib
from typing import Protocol

from PIL import Image, ImageDraw


class ImageBackend(Protocol):
    name: str
    model: str
    revision: str

    def render(self, prompt: str, width: int, height: int, seed: int) -> Image.Image: ...


class FakeBackend:
    """Deterministic placeholder: the same prompt, size and seed always give the same pixels."""

    name = "fake"
    model = "chriek-fake-backend"
    revision = "fake-v1"

    def render(self, prompt: str, width: int, height: int, seed: int) -> Image.Image:
        digest = hashlib.sha256(f"{prompt}\n{seed}\n{width}x{height}".encode()).digest()
        image = Image.new("RGB", (width, height), tuple(digest[0:3]))
        inset_x = width * (1 + digest[6] % 3) // 8
        inset_y = height * (1 + digest[7] % 3) // 8
        ImageDraw.Draw(image).rectangle(
            (inset_x, inset_y, width - inset_x - 1, height - inset_y - 1), fill=tuple(digest[3:6])
        )
        return image
