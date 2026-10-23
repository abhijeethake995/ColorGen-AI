"""Environment-backed configuration for the DDColor-T Flask application."""
from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    ddcolor_checkpoint_path: Path = Path(os.getenv(
        "COLORGEN_DDCOLOR_CHECKPOINT", ROOT / "models" / "ddcolor_paper_tiny.bin"
    ))
    ddcolor_input_size: int = int(os.getenv("COLORGEN_DDCOLOR_INPUT_SIZE", "512"))
    max_upload_bytes: int = int(os.getenv("COLORGEN_MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
    max_image_pixels: int = int(os.getenv("COLORGEN_MAX_IMAGE_PIXELS", "20000000"))
    device: str = os.getenv("COLORGEN_DEVICE", "auto")


settings = Settings()


def resolve_project_path(path: Path) -> Path:
    """Resolve relative configuration paths from the project root."""
    return path if path.is_absolute() else ROOT / path
