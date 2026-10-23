from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from backend.app import create_app
from backend.config import settings
from backend.image_utils import supported_upload_formats


def png_bytes(size=(47, 31), color=(120, 70, 180)):
    buffer = BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


class FixedColorizer(torch.nn.Module):
    """API mechanics test double; it is never used as a production model."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, image):
        return torch.full((1, 2, image.shape[2], image.shape[3]), 18.0, device=image.device)


def app_with_test_model(monkeypatch, tmp_path, **config):
    weights = tmp_path / "test-ddcolor.bin"
    weights.write_bytes(b"unit-test-only-placeholder")
    monkeypatch.setattr("backend.app.load_ddcolor", lambda *_args, **_kwargs: FixedColorizer())
    return create_app({"TESTING": True, "DEVICE": "cpu", "CHECKPOINT_PATH": weights, **config})


def test_health_and_model_info_report_ddcolor_as_the_only_engine(tmp_path):
    app = create_app({"TESTING": True, "DEVICE": "cpu", "CHECKPOINT_PATH": tmp_path / "missing.bin"})
    client = app.test_client()
    health = client.get("/api/health").get_json()
    info = client.get("/api/model-info").get_json()

    assert health["status"] == "ok" and health["checkpoint_available"] is False
    assert health["device"] == "cpu" and health["max_upload_bytes"] > 0
    assert health["model_engine"] == info["model_engine"] == "ddcolor"
    assert {item["format"] for item in health["supported_formats"]} >= {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF"}
    assert "DDColor-T" in health["model_config"]["architecture"]
    assert "DDColor-T" in info["architecture"]
    assert info["checkpoint_available"] is False and info["validation_metrics"] is None

    response = client.post("/api/colorize", data={"image": (png_bytes(), "test.png")},
                           content_type="multipart/form-data")
    assert response.status_code == 503
    assert response.get_json()["error"]["code"] == "checkpoint_unavailable"
    assert "DDColor weights are missing" in response.get_json()["error"]["message"]


def test_active_loader_has_no_legacy_engine_or_checkpoint_setting():
    root = Path(__file__).resolve().parents[1]
    application_source = (root / "backend" / "app.py").read_text(encoding="utf-8")
    config_source = (root / "backend" / "config.py").read_text(encoding="utf-8")
    assert not hasattr(settings, "engine")
    for legacy_reference in ("COLORGEN_ENGINE", "COLORGEN_CHECKPOINT", "best_model.pt", "build_generator"):
        assert legacy_reference not in application_source
        assert legacy_reference not in config_source


def test_invalid_ddcolor_checkpoint_reports_clear_startup_error(tmp_path):
    weights = tmp_path / "corrupt.bin"
    weights.write_bytes(b"not a PyTorch checkpoint")
    client = create_app({"TESTING": True, "DEVICE": "cpu", "CHECKPOINT_PATH": weights}).test_client()
    health = client.get("/api/health").get_json()
    assert not health["checkpoint_available"]
    assert "DDColor checkpoint could not be loaded" in health["model_error"]


def test_api_rejects_unsupported_decoded_format(monkeypatch, tmp_path):
    client = app_with_test_model(monkeypatch, tmp_path).test_client()
    ppm = BytesIO(b"P6\n1 1\n255\n\xff\x00\x00")
    response = client.post("/api/colorize", data={"image": (ppm, "x.png", "image/png")},
                           content_type="multipart/form-data")
    assert response.status_code == 415
    assert response.get_json()["error"]["code"] == "unsupported_type"


@pytest.mark.parametrize(("image_format", "filename", "mime"), [
    ("JPEG", "photo.jpg", "image/jpeg"),
    ("PNG", "photo.png", "image/png"),
    ("WEBP", "photo.webp", "image/webp"),
    ("BMP", "photo.bmp", "image/bmp"),
    ("TIFF", "photo.tiff", "image/tiff"),
    ("GIF", "photo.gif", "image/gif"),
])
def test_supported_format_uploads_colorize_to_original_size(monkeypatch, tmp_path, image_format, filename, mime):
    if image_format not in {item["format"] for item in supported_upload_formats()}:
        pytest.skip(f"No {image_format} decoder is installed")
    client = app_with_test_model(monkeypatch, tmp_path).test_client()
    source = BytesIO()
    Image.new("RGB", (47, 31), color=(120, 70, 180)).save(source, format=image_format)
    source.seek(0)
    response = client.post("/api/colorize", data={"image": (source, filename, mime)},
                           content_type="multipart/form-data")
    assert response.status_code == 200 and response.mimetype == "image/png"
    assert response.headers["X-Image-Width"] == "47"
    assert response.headers["X-Image-Height"] == "31"
    with Image.open(BytesIO(response.data)) as result:
        assert result.mode == "RGB" and result.size == (47, 31)


def test_api_uses_decoded_content_instead_of_mime_or_extension(monkeypatch, tmp_path):
    client = app_with_test_model(monkeypatch, tmp_path).test_client()
    response = client.post(
        "/api/colorize",
        data={"image": (png_bytes(), "disguised.gif", "application/octet-stream")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200 and response.mimetype == "image/png"


@pytest.mark.parametrize(("image_format", "filename", "mime"), [
    ("AVIF", "photo.avif", "image/avif"),
    ("HEIF", "photo.heic", "image/heic"),
])
def test_optional_avif_heif_api_upload_when_decoder_is_available(monkeypatch, tmp_path, image_format, filename, mime):
    if image_format not in {item["format"] for item in supported_upload_formats()}:
        pytest.skip(f"No {image_format} decoder is installed")
    source = BytesIO()
    try:
        Image.new("RGB", (47, 31), color=(120, 70, 180)).save(source, format=image_format)
    except (OSError, ValueError) as exc:
        pytest.skip(f"Installed {image_format} decoder cannot encode test fixtures: {exc}")
    source.seek(0)
    client = app_with_test_model(monkeypatch, tmp_path).test_client()
    response = client.post("/api/colorize", data={"image": (source, filename, mime)},
                           content_type="multipart/form-data")
    assert response.status_code == 200 and response.mimetype == "image/png"
    assert response.headers["X-Image-Width"] == "47"
    assert response.headers["X-Image-Height"] == "31"


def test_corrupt_image_returns_structured_error(monkeypatch, tmp_path):
    client = app_with_test_model(monkeypatch, tmp_path).test_client()
    response = client.post("/api/colorize", data={"image": (BytesIO(b"not a valid image"), "broken.png")},
                           content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_image"


def test_upload_size_limit(monkeypatch, tmp_path):
    client = app_with_test_model(monkeypatch, tmp_path, MAX_UPLOAD_BYTES=256).test_client()
    response = client.post("/api/colorize", data={"image": (BytesIO(b"x" * 1024), "large.png")},
                           content_type="multipart/form-data")
    assert response.status_code == 413


def test_excessive_image_dimensions_are_rejected(monkeypatch, tmp_path):
    client = app_with_test_model(monkeypatch, tmp_path, MAX_IMAGE_PIXELS=100).test_client()
    source = png_bytes(size=(11, 10))
    response = client.post("/api/colorize", data={"image": (source, "large-dimensions.png")},
                           content_type="multipart/form-data")
    assert response.status_code == 413
    assert response.get_json()["error"]["code"] == "image_too_large"


def test_inference_failure_does_not_expose_internal_exception(monkeypatch, tmp_path):
    client = app_with_test_model(monkeypatch, tmp_path).test_client()

    def fail_inference(*_args, **_kwargs):
        raise RuntimeError("private implementation detail")

    monkeypatch.setattr("backend.app.colorize_rgb", fail_inference)
    response = client.post("/api/colorize", data={"image": (png_bytes(), "test.png")},
                           content_type="multipart/form-data")
    assert response.status_code == 500
    message = response.get_json()["error"]["message"]
    assert "Colorization failed" in message
    assert "private implementation detail" not in message


def test_successful_ddcolor_api_response_is_rgb_png_with_original_dimensions(monkeypatch, tmp_path):
    client = app_with_test_model(monkeypatch, tmp_path).test_client()
    response = client.post("/api/colorize", data={"image": (png_bytes(), "test.png")},
                           content_type="multipart/form-data")

    assert response.status_code == 200 and response.mimetype == "image/png"
    assert float(response.headers["X-Inference-Time-Ms"]) > 0
    assert response.headers["X-Image-Width"] == "47"
    assert response.headers["X-Image-Height"] == "31"
    with Image.open(BytesIO(response.data)) as result:
        assert result.mode == "RGB" and result.size == (47, 31)
        pixels = np.asarray(result)
        assert np.any(np.ptp(pixels.astype(np.int16), axis=2) > 5)


def test_frontend_entry_retains_upload_result_and_download_controls(tmp_path):
    client = create_app({"TESTING": True, "DEVICE": "cpu", "CHECKPOINT_PATH": tmp_path / "missing.bin"}).test_client()
    page = client.get("/").get_data(as_text=True)
    script = Path(__file__).resolve().parents[1] / "frontend" / "script.js"
    javascript = script.read_text(encoding="utf-8")

    for control_id in ("fileInput", "dropzone", "browseButton", "colorizeButton", "results",
                       "compareView", "sideView", "downloadButton", "anotherButton"):
        assert f'id="{control_id}"' in page
    assert "fetch('/api/colorize'" in javascript
    assert "health.supported_formats" in javascript and "fileInput.accept" in javascript
    assert "URL.createObjectURL" in javascript and "URL.revokeObjectURL" in javascript
    assert "first frame" in page
    assert "PRETRAINED RESEARCH MODEL" not in page
    assert "A LITTLE BIT OF MAGIC" not in page
