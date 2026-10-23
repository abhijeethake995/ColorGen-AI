"""Download the pinned DDColor-T checkpoint with atomic SHA-256 verification.

This command is used by the Docker image build. The normal local Flask startup
does not call it and continues using the checkpoint already on disk.
"""
import argparse
import hashlib
from http.client import HTTPException
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from .model_artifact import MODEL_SHA256, MODEL_URL

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(
    destination,
    *,
    opener=None,
    expected_sha256=None,
    url=None,
    retries=3,
    retry_delay=2,
):
    """Download once, verify before atomically installing, and preserve bad files.

    Returns ``False`` when an already verified file was reused and ``True``
    after a successful download. An existing file with a wrong hash is never
    silently replaced.
    """
    destination = Path(destination)
    expected_sha256 = (expected_sha256 or MODEL_SHA256).lower()
    url = url or MODEL_URL
    opener = opener or urlopen

    if destination.is_file():
        actual = sha256_file(destination)
        if actual == expected_sha256:
            return False
        raise ValueError(
            f"Existing DDColor checkpoint at {destination} failed SHA-256 verification."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(retries):
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
        )
        temporary_path = Path(temporary_name)
        digest = hashlib.sha256()
        try:
            request = Request(url, headers={"User-Agent": "ColorGen-AI/1.0"})
            with os.fdopen(fd, "wb") as output, opener(request, timeout=90) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())
        except (OSError, TimeoutError, URLError, HTTPException) as exc:
            last_error = exc
            temporary_path.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(retry_delay * (attempt + 1))
            continue
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        actual = digest.hexdigest()
        if actual != expected_sha256:
            temporary_path.unlink(missing_ok=True)
            raise ValueError(
                "Downloaded DDColor checkpoint failed SHA-256 verification "
                f"(expected {expected_sha256}, received {actual})."
            )

        try:
            os.replace(temporary_path, destination)
        except OSError:
            temporary_path.unlink(missing_ok=True)
            raise
        return True

    raise RuntimeError(
        f"Unable to download the official DDColor checkpoint after {retries} attempts: {last_error}"
    ) from last_error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "ddcolor_paper_tiny.bin",
        help="Checkpoint destination path.",
    )
    args = parser.parse_args(argv)
    try:
        downloaded = download_checkpoint(args.output)
    except Exception as exc:
        print(f"DDColor checkpoint setup failed: {exc}", file=sys.stderr)
        return 1
    action = "Downloaded and verified" if downloaded else "Reused verified"
    print(f"{action} DDColor checkpoint at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
