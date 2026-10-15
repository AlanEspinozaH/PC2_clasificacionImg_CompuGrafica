"""Decodificación compartida entre captura e inferencia."""

import base64
import binascii
import io

from PIL import Image, UnidentifiedImageError

from .preprocessing import EmptyDrawingError, preprocess_image


def decode_drawing(data_url, image_size=32):
    prefix = "data:image/png;base64,"
    if not isinstance(data_url, str) or not data_url.startswith(prefix):
        raise ValueError("El dibujo debe enviarse como una imagen PNG.")
    try:
        decoded = base64.b64decode(data_url[len(prefix):], validate=True)
        if len(decoded) > 1024 * 1024:
            raise ValueError("La imagen supera el tamaño permitido.")
        with Image.open(io.BytesIO(decoded)) as image:
            if image.format != "PNG" or max(image.size) > 1024:
                raise ValueError("Usa un PNG de hasta 1024 píxeles por lado.")
            image.load()
            processed = preprocess_image(image, size=image_size)
            return image.copy(), processed
    except EmptyDrawingError:
        raise
    except (binascii.Error, UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        raise ValueError("No se pudo leer el PNG. Vuelve a dibujar el símbolo.") from error
