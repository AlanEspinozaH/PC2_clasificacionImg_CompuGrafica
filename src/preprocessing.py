"""Mismo procesamiento para los datos y la futura predicción."""

import numpy as np
from PIL import Image, ImageOps


class EmptyDrawingError(ValueError):
    """El lienzo no contiene tinta suficiente para guardar una muestra."""


def flatten_on_white(image):
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return ImageOps.grayscale(Image.alpha_composite(white, rgba))


def preprocess_image(image, size=32, padding=4):
    """Devuelve (size, size, 1): fondo 0, tinta 1, valores float32."""
    if size <= 2 * padding:
        raise ValueError("El tamaño debe ser mayor que el margen total.")
    ink = ImageOps.invert(flatten_on_white(image))
    mask = np.asarray(ink) > 20
    if np.count_nonzero(mask) < 8:
        raise EmptyDrawingError("Dibuja el símbolo antes de guardarlo.")
    bounding_box = Image.fromarray(mask.astype(np.uint8) * 255).getbbox()
    cropped = ink.crop(bounding_box)
    fitted = ImageOps.contain(
        cropped, (size - 2 * padding, size - 2 * padding), Image.Resampling.LANCZOS
    )
    centered = Image.new("L", (size, size), 0)
    position = ((size - fitted.width) // 2, (size - fitted.height) // 2)
    centered.paste(fitted, position)
    return (np.asarray(centered, dtype=np.float32) / 255.0)[..., np.newaxis]
