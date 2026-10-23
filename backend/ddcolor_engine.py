"""DDColor model loading and image pipeline.

The architecture code in ``backend.vendor`` is vendored from the official
Apache-2.0 DDColor repository. Input and output handling follows its published
inference pipeline, with aspect-preserving letterboxing added for uploads.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from .vendor.ddcolor.model import DDColor
from .model_artifact import MODEL_ID, MODEL_REVISION, MODEL_SHA256


DEFAULT_INPUT_SIZE = 512


def build_ddcolor(input_size=DEFAULT_INPUT_SIZE):
    """Build the architecture using the publisher's tiny-model configuration."""
    return DDColor(
        encoder_name="convnext-t",
        decoder_name="MultiScaleColorDecoder",
        num_input_channels=3,
        input_size=(input_size, input_size),
        nf=512,
        num_output_channels=2,
        last_norm="Spectral",
        do_normalize=False,
        num_queries=100,
        num_scales=3,
        dec_layers=9,
    )


def load_ddcolor(checkpoint_path, device="cpu", input_size=DEFAULT_INPUT_SIZE):
    """Load the official state dict with restricted deserialization and strict keys."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"DDColor weights are missing at {path}. Download the official "
            f"{MODEL_ID} checkpoint using the README setup steps."
        )

    # Load tensor storage lazily from the checkpoint, then assign those exact
    # tensors to the model. This avoids holding a second 220 MB copy during
    # startup on memory-constrained CPU hosts. The official checkpoint is a
    # seekable local file, and mmap/assign are supported by our PyTorch minimum.
    # weights_only=True blocks arbitrary pickle globals while loading tensors.
    state_dict = torch.load(
        path, map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("DDColor checkpoint must be a non-empty state dictionary.")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor)
               for key, value in state_dict.items()):
        raise ValueError("DDColor checkpoint contains non-tensor state entries.")

    model = build_ddcolor(input_size)
    model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    model.eval().to(device)
    return model


def colorize_rgb(model, rgb: np.ndarray, input_size=DEFAULT_INPUT_SIZE):
    """Colorize one RGB uint8 image and return an RGB uint8 image of same size.

    DDColor expects a grayscale image represented as RGB floats in [0, 1]; the
    model applies its own ImageNet normalization. It predicts OpenCV LAB a/b
    values. The uploaded image's original L plane is preserved for reconstruction.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 image.")
    height, width = rgb.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("Image dimensions must be positive.")

    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    original_l = lab[:, :, 0]

    scale = min(input_size / width, input_size / height)
    resized_w = max(1, min(input_size, round(width * scale)))
    resized_h = max(1, min(input_size, round(height * scale)))
    left = (input_size - resized_w) // 2
    top = (input_size - resized_h) // 2
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized_l = cv2.resize(original_l, (resized_w, resized_h), interpolation=interpolation)

    # Neutral mid-gray padding gives the network valid, non-color context.
    letterboxed_l = np.full((input_size, input_size), 50.0, dtype=np.float32)
    letterboxed_l[top:top + resized_h, left:left + resized_w] = resized_l
    gray_lab = np.zeros((input_size, input_size, 3), dtype=np.float32)
    gray_lab[:, :, 0] = letterboxed_l
    gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)
    tensor = torch.from_numpy(gray_rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    tensor = tensor.to(device)
    with torch.inference_mode():
        predicted_ab = model(tensor)
    if predicted_ab.ndim != 4 or predicted_ab.shape[:2] != (1, 2):
        raise ValueError(f"DDColor returned unexpected output shape {tuple(predicted_ab.shape)}.")
    predicted_ab = predicted_ab[0].float().cpu().numpy().transpose(1, 2, 0)
    predicted_ab = predicted_ab[top:top + resized_h, left:left + resized_w]
    predicted_ab = cv2.resize(predicted_ab, (width, height), interpolation=cv2.INTER_CUBIC)

    output_lab = np.empty((height, width, 3), dtype=np.float32)
    output_lab[:, :, 0] = original_l
    output_lab[:, :, 1:] = predicted_ab
    output_rgb = cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB)
    return np.clip(output_rgb * 255.0, 0, 255).round().astype(np.uint8)
