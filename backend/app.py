"""Flask application for ColorGen AI. DDColor-T is the only supported model."""
from io import BytesIO
from pathlib import Path
import time

import torch
from flask import Flask, jsonify, request, send_file

from .config import settings, resolve_project_path
from .ddcolor_engine import DEFAULT_INPUT_SIZE, MODEL_ID, colorize_rgb, load_ddcolor
from .image_utils import (
    ImageDimensionsTooLarge,
    UnsupportedImageFormat,
    decode_image,
    supported_upload_formats,
)


def _device(preference=None):
    preference = preference or settings.device
    if preference != "auto":
        try:
            requested = torch.device(preference)
        except (RuntimeError, ValueError):
            return torch.device("cpu")
        if requested.type not in {"cpu", "cuda"}:
            return torch.device("cpu")
        if requested.type == "cuda" and (not torch.cuda.is_available() or
                                             (requested.index is not None and requested.index >= torch.cuda.device_count())):
            return torch.device("cpu")
        return requested
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(path=None, device_preference=None, input_size=DEFAULT_INPUT_SIZE):
    path = resolve_project_path(Path(path or settings.ddcolor_checkpoint_path))
    if not path.is_file():
        return None, None, (
            f"DDColor weights are missing at {path}. Download the official "
            "checkpoint using the model setup steps in README.md."
        )

    device = _device(device_preference)
    try:
        model = load_ddcolor(path, device=device, input_size=input_size)
        metadata = {
            "model_id": MODEL_ID,
            "input_size": input_size,
            "architecture": "DDColor-T with ConvNeXt-T encoder and dual decoders",
            "weights": str(path),
        }
        return model, metadata, None
    except Exception as exc:
        return None, None, f"DDColor checkpoint could not be loaded: {exc}"


def create_app(test_config=None):
    app = Flask(__name__, static_folder="../frontend", static_url_path="")
    app.config["MAX_UPLOAD_BYTES"] = settings.max_upload_bytes
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_bytes + 64 * 1024  # Multipart headers and framing.
    app.config["MAX_IMAGE_PIXELS"] = settings.max_image_pixels
    if test_config:
        app.config.update(test_config)

    checkpoint_path = app.config.get("CHECKPOINT_PATH", settings.ddcolor_checkpoint_path)
    device_preference = app.config.get("DEVICE")
    input_size = int(app.config.get("DDCOLOR_INPUT_SIZE", settings.ddcolor_input_size))
    model, checkpoint, model_error = _load_model(checkpoint_path, device_preference, input_size)
    device = _device(device_preference)
    app.extensions["colorgen"] = {
        "model": model, "checkpoint": checkpoint, "error": model_error,
        "device": device, "input_size": input_size,
    }

    @app.errorhandler(413)
    def too_large(_error):
        limit = app.config.get("MAX_UPLOAD_BYTES", settings.max_upload_bytes)
        return jsonify(error={"code": "upload_too_large", "message": f"File exceeds the {limit / (1024 * 1024):.1f} MiB upload limit."}), 413

    @app.get("/")
    def home():
        return app.send_static_file("index.html")

    @app.get("/api/health")
    def health():
        state = app.extensions["colorgen"]
        checkpoint_data = state["checkpoint"] or {}
        return jsonify(status="ok", checkpoint_available=state["model"] is not None,
                       model_error=state["error"], device=str(state["device"]),
                       max_upload_bytes=app.config.get("MAX_UPLOAD_BYTES", settings.max_upload_bytes),
                       max_image_pixels=app.config.get("MAX_IMAGE_PIXELS", settings.max_image_pixels),
                       supported_formats=supported_upload_formats(),
                       model_engine="ddcolor",
                       model_config={
                           "input_resolution": checkpoint_data.get("input_size", state["input_size"]),
                           "architecture": checkpoint_data.get(
                               "architecture", "DDColor-T with ConvNeXt-T encoder and dual decoders"
                           ),
                       })

    @app.get("/api/model-info")
    def model_info():
        state = app.extensions["colorgen"]
        checkpoint_data = state["checkpoint"] or {}
        return jsonify(architecture=checkpoint_data.get(
                           "architecture", "DDColor-T with ConvNeXt-T encoder and dual decoders"),
                       input="Grayscale RGB in [0, 1], ImageNet-normalized inside the model",
                       output="LAB chrominance a/b", model_engine="ddcolor",
                       input_resolution=checkpoint_data.get("input_size", state["input_size"]),
                       checkpoint_available=state["model"] is not None, model_error=state["error"],
                       training_stage=None, validation_metrics=None,
                       pretrained_model_id=checkpoint_data.get("model_id", MODEL_ID))

    @app.post("/api/colorize")
    def colorize():
        state = app.extensions["colorgen"]
        if state["model"] is None:
            return jsonify(error={"code": "checkpoint_unavailable", "message": state["error"]}), 503
        upload = request.files.get("image")
        if upload is None:
            return jsonify(error={"code": "missing_file", "message": "Add an image in the 'image' form field."}), 400
        upload_limit = app.config.get("MAX_UPLOAD_BYTES", settings.max_upload_bytes)
        data = upload.stream.read(upload_limit + 1)
        if len(data) > upload_limit:
            return jsonify(error={"code": "upload_too_large", "message": "Image exceeds the upload size limit."}), 413
        try:
            rgb = decode_image(data, max_pixels=app.config.get("MAX_IMAGE_PIXELS", settings.max_image_pixels))
        except UnsupportedImageFormat as exc:
            return jsonify(error={"code": "unsupported_type", "message": str(exc)}), 415
        except ImageDimensionsTooLarge as exc:
            return jsonify(error={"code": "image_too_large", "message": str(exc)}), 413
        except ValueError as exc:
            return jsonify(error={"code": "invalid_image", "message": str(exc)}), 400

        start = time.perf_counter()
        try:
            result = colorize_rgb(state["model"], rgb, input_size=state["input_size"])
            original_size = (rgb.shape[1], rgb.shape[0])
        except Exception:
            app.logger.exception("Colorization inference failed")
            return jsonify(error={"code": "inference_failed", "message": "Colorization failed. Please try another image or retry."}), 500

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        output = BytesIO()
        from PIL import Image
        Image.fromarray(result).save(output, format="PNG")
        output.seek(0)
        response = send_file(output, mimetype="image/png", as_attachment=False, download_name="colorgen-colorized.png")
        response.headers["X-Inference-Time-Ms"] = f"{elapsed_ms:.2f}"
        response.headers["X-Image-Width"] = str(original_size[0])
        response.headers["X-Image-Height"] = str(original_size[1])
        response.headers["Access-Control-Expose-Headers"] = "X-Inference-Time-Ms, X-Image-Width, X-Image-Height"
        return response
    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(__import__("os").getenv("PORT", "5000")), debug=False)
