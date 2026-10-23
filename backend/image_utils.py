"""Safe decoding of uploaded raster images into orientation-corrected RGB arrays."""
from io import BytesIO
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


# Register the optional HEIF decoder when installed. Pillow 11.3+ may include
# AVIF support directly; pillow-avif-plugin is also detected when installed.
try:  # pragma: no cover - availability depends on the user's environment
    import pillow_heif

    pillow_heif.register_heif_opener()
except (ImportError, OSError):  # Optional decoder may lack its native library.
    pass

try:  # pragma: no cover - availability depends on the user's environment
    import pillow_avif  # noqa: F401
except (ImportError, OSError):  # Optional decoder may lack its native library.
    pass


FORMAT_DETAILS = {
    "JPEG": {"label": "JPG/JPEG", "extensions": [".jpg", ".jpeg"], "mime_types": ["image/jpeg"]},
    "PNG": {"label": "PNG", "extensions": [".png"], "mime_types": ["image/png"]},
    "WEBP": {"label": "WebP", "extensions": [".webp"], "mime_types": ["image/webp"]},
    "BMP": {"label": "BMP", "extensions": [".bmp"], "mime_types": ["image/bmp"]},
    "TIFF": {"label": "TIFF/TIF", "extensions": [".tif", ".tiff"], "mime_types": ["image/tiff"]},
    "GIF": {"label": "GIF", "extensions": [".gif"], "mime_types": ["image/gif"]},
    "AVIF": {"label": "AVIF", "extensions": [".avif"], "mime_types": ["image/avif"]},
    "HEIF": {"label": "HEIC/HEIF", "extensions": [".heic", ".heif"], "mime_types": ["image/heic", "image/heif"]},
    "HEIC": {"label": "HEIC/HEIF", "extensions": [".heic", ".heif"], "mime_types": ["image/heic", "image/heif"]},
}


class UnsupportedImageFormat(ValueError):
    """Raised when Pillow has decoded a format the app does not support."""


class ImageDimensionsTooLarge(ValueError):
    """Raised when an image exceeds the configured pixel limit."""


def supported_upload_formats():
    """Return formats supported by decoders actually registered in Pillow."""
    registered = set(Image.registered_extensions().values())
    return [
        {"format": image_format, **FORMAT_DETAILS[image_format]}
        for image_format in FORMAT_DETAILS
        if image_format in registered
    ]


def _check_dimensions(image, max_pixels):
    if image.width <= 0 or image.height <= 0 or image.width * image.height > max_pixels:
        raise ImageDimensionsTooLarge(
            f"Image dimensions exceed the {max_pixels:,}-pixel processing limit."
        )


def decode_image(data: bytes, max_pixels=20_000_000) -> np.ndarray:
    """Decode actual image bytes to RGB uint8, compositing transparency on white.

    Animated GIF and WebP uploads use their first frame. File extensions and
    multipart MIME types are intentionally ignored; Pillow identifies the bytes.
    """
    if not data:
        raise ValueError("The uploaded file is empty.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as probe:
                image_format = probe.format
                if image_format not in {item["format"] for item in supported_upload_formats()}:
                    raise UnsupportedImageFormat(
                        "Unsupported image format. Use one of the formats listed by the upload area."
                    )
                _check_dimensions(probe, max_pixels)
                probe.verify()

            with Image.open(BytesIO(data)) as image:
                _check_dimensions(image, max_pixels)
                # Animated formats intentionally use their first frame.
                image.seek(0)
                image.load()
                image = ImageOps.exif_transpose(image)
                if "A" in image.getbands() or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                    image = Image.alpha_composite(background, rgba).convert("RGB")
                else:
                    image = image.convert("RGB")
                return np.asarray(image, dtype=np.uint8)
    except (UnsupportedImageFormat, ImageDimensionsTooLarge):
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ImageDimensionsTooLarge(
            "Image dimensions are too large to process safely."
        ) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ValueError("The uploaded image is corrupt or could not be decoded.") from exc
