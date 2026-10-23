from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from backend.image_utils import (
    ImageDimensionsTooLarge,
    UnsupportedImageFormat,
    decode_image,
    supported_upload_formats,
)


def encoded(mode="RGB", size=(73, 51), image_format="PNG"):
    buffer = BytesIO()
    color = {"RGB": (120, 45, 200), "RGBA": (120, 45, 200, 96), "L": 120}[mode]
    Image.new(mode, size, color=color).save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_decode_common_modes_to_rgb_uint8(mode):
    result = decode_image(encoded(mode))
    assert result.shape == (51, 73, 3) and result.dtype == np.uint8


@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF"])
def test_common_raster_upload_formats_decode(image_format):
    result = decode_image(encoded(image_format=image_format))
    assert result.shape == (51, 73, 3) and result.dtype == np.uint8


@pytest.mark.parametrize("image_format", ["AVIF", "HEIF"])
def test_optional_avif_and_heif_when_decoder_is_available(image_format):
    if image_format not in {item["format"] for item in supported_upload_formats()}:
        pytest.skip(f"No {image_format} decoder is installed")
    try:
        data = encoded(image_format=image_format)
    except (OSError, ValueError) as exc:
        pytest.skip(f"Installed {image_format} decoder cannot encode test fixtures: {exc}")
    result = decode_image(data)
    assert result.shape == (51, 73, 3)


def test_transparent_png_is_composited_onto_white():
    result = decode_image(encoded(mode="RGBA"))
    assert np.all(result[0, 0] > np.array([200, 170, 220], dtype=np.uint8))


def test_animated_gif_uses_its_first_frame():
    buffer = BytesIO()
    first = Image.new("RGB", (12, 8), (220, 20, 30))
    second = Image.new("RGB", (12, 8), (10, 30, 220))
    first.save(buffer, format="GIF", save_all=True, append_images=[second], duration=100, loop=0)

    result = decode_image(buffer.getvalue())
    assert result.shape == (8, 12, 3)
    assert result[0, 0, 0] > result[0, 0, 2]


def test_unsupported_decoded_format_has_specific_error():
    with pytest.raises(UnsupportedImageFormat, match="Unsupported image format"):
        decode_image(b"P6\n1 1\n255\n\xff\x00\x00")


def test_invalid_and_corrupt_images_raise_clear_errors():
    with pytest.raises(ValueError, match="corrupt"):
        decode_image(b"not an image")


def test_oversized_pixel_count_is_rejected_before_decode():
    with pytest.raises(ImageDimensionsTooLarge, match="pixel processing limit"):
        decode_image(encoded(size=(20, 20)), max_pixels=100)
