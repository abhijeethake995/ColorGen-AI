"""DDColor preprocessing, loading safety, and real checkpoint integration."""
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from backend.app import app as default_app
from backend.ddcolor_engine import (
    DEFAULT_INPUT_SIZE,
    colorize_rgb,
    load_ddcolor,
)


ROOT = Path(__file__).resolve().parents[1]
DDCOLOR_WEIGHTS = ROOT / "models" / "ddcolor_paper_tiny.bin"


class FixedChroma(torch.nn.Module):
    """Small preprocessing test double. Real integration is tested separately."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.seen = None

    def forward(self, image):
        self.seen = image.detach().cpu()
        return torch.zeros((1, 2, image.shape[2], image.shape[3]), device=image.device) + 18


def test_ddcolor_input_is_grayscale_rgb_unit_range_and_aspect_preserved():
    model = FixedChroma()
    rgb = np.zeros((47, 93, 3), dtype=np.uint8)
    rgb[:] = (190, 75, 30)
    result = colorize_rgb(model, rgb, input_size=128)

    assert result.shape == rgb.shape and result.dtype == np.uint8
    assert model.seen.shape == (1, 3, 128, 128)
    assert model.seen.min() >= 0 and model.seen.max() <= 1
    np.testing.assert_allclose(model.seen[:, 0], model.seen[:, 1], atol=1e-6)
    np.testing.assert_allclose(model.seen[:, 1], model.seen[:, 2], atol=1e-6)
    assert np.any(np.ptp(result.astype(np.int16), axis=2) > 5)


def test_ddcolor_missing_checkpoint_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="DDColor weights are missing"):
        load_ddcolor(tmp_path / "missing.bin", input_size=128)


@pytest.mark.skipif(not DDCOLOR_WEIGHTS.is_file(), reason="Official DDColor checkpoint is not installed.")
def test_official_ddcolor_weights_load_strictly_and_have_expected_architecture():
    model = load_ddcolor(DDCOLOR_WEIGHTS, device="cpu", input_size=DEFAULT_INPUT_SIZE)
    assert sum(parameter.numel() for parameter in model.parameters()) == 55_006_640
    assert next(model.parameters()).device.type == "cpu"


@pytest.mark.skipif(not DDCOLOR_WEIGHTS.is_file(), reason="Official DDColor checkpoint is not installed.")
def test_real_ddcolor_api_returns_colorized_png_at_original_dimensions():
    state = default_app.extensions["colorgen"]
    if state["model"] is None:
        pytest.skip(f"Default DDColor model is unavailable: {state['error']}")

    # Real cat photo from scikit-image. The reference RGB pixels are not passed
    # to the model; only this grayscale upload goes through the HTTP endpoint.
    skimage_data = pytest.importorskip("skimage.data")
    grayscale = Image.fromarray(skimage_data.chelsea()).convert("L")
    buffer = BytesIO()
    grayscale.save(buffer, format="PNG")
    buffer.seek(0)

    response = default_app.test_client().post(
        "/api/colorize",
        data={"image": (buffer, "cat-grayscale.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.mimetype == "image/png"
    assert float(response.headers["X-Inference-Time-Ms"]) > 0
    with Image.open(BytesIO(response.data)) as colorized:
        assert colorized.mode == "RGB"
        assert colorized.size == grayscale.size
        pixels = np.asarray(colorized)
        assert np.mean(np.ptp(pixels.astype(np.int16), axis=2) > 5) > 0.20


def test_invalid_checkpoint_is_rejected(tmp_path):
    checkpoint = tmp_path / "invalid.bin"
    checkpoint.write_bytes(b"not a state dictionary")
    with pytest.raises(Exception):
        load_ddcolor(checkpoint, device="cpu", input_size=128)
