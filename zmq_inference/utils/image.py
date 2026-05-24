"""Image encode/decode helpers for the ZMQ inference pipeline."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Tuple, Union

from PIL import Image


def encode_image(source: Union[str, Path, Image.Image]) -> str:
    """Encode an image to a base64 string.

    Args:
        source: A filesystem path (``str`` or :class:`pathlib.Path`) or an
                already-loaded :class:`PIL.Image.Image` object.

    Returns:
        A base64-encoded JPEG string.

    Raises:
        FileNotFoundError: If *source* is a path that does not exist.
        TypeError: If *source* is not a recognised type.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")

    if isinstance(source, Image.Image):
        buf = io.BytesIO()
        # Normalise mode to RGB before JPEG encoding (JPEG does not support alpha).
        img = source.convert("RGB") if source.mode != "RGB" else source
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    raise TypeError(
        f"encode_image expects a path (str/Path) or PIL.Image.Image, got {type(source).__name__}"
    )


def decode_image(b64_str: str) -> Image.Image:
    """Decode a base64 image string back to a PIL Image.

    Args:
        b64_str: A base64-encoded image string (any PIL-supported format).

    Returns:
        A :class:`PIL.Image.Image` opened in RGB mode.
    """
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw))
    return img.convert("RGB")


def preprocess_for_display(
    img: Image.Image,
    size: Tuple[int, int] = (224, 224),
) -> Image.Image:
    """Resize an image for display or debugging purposes.

    This is a lightweight helper intended for visualisation only; it does
    *not* apply any model-specific normalisation.

    Args:
        img:  Source image.
        size: Target ``(width, height)`` tuple.

    Returns:
        A resized :class:`PIL.Image.Image` in RGB mode.
    """
    return img.convert("RGB").resize(size, Image.BILINEAR)
