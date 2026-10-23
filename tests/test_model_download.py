from hashlib import sha256
from io import BytesIO

import pytest

import backend.model_download as downloader


class FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_valid_local_checkpoint_is_reused_without_network(tmp_path):
    content = b"verified checkpoint fixture"
    target = tmp_path / "weights.bin"
    target.write_bytes(content)

    def network_must_not_be_used(*_args, **_kwargs):
        raise AssertionError("download attempted despite valid local checkpoint")

    assert downloader.download_checkpoint(
        target, opener=network_must_not_be_used, expected_sha256=sha256(content).hexdigest()
    ) is False
    assert target.read_bytes() == content


def test_download_verifies_checksum_and_atomically_installs_file(tmp_path):
    content = b"new checkpoint fixture" * 100
    target = tmp_path / "nested" / "weights.bin"
    calls = []

    def fake_open(request, timeout):
        calls.append((request.full_url, timeout))
        return FakeResponse(content)

    result = downloader.download_checkpoint(
        target,
        opener=fake_open,
        expected_sha256=sha256(content).hexdigest(),
        url="https://example.invalid/pinned.bin",
    )

    assert result is True
    assert target.read_bytes() == content
    assert calls == [("https://example.invalid/pinned.bin", 90)]
    assert list(target.parent.glob("*.part")) == []


def test_checksum_mismatch_leaves_no_final_or_temporary_file(tmp_path):
    target = tmp_path / "weights.bin"
    content = b"not the expected checkpoint"

    with pytest.raises(ValueError, match="failed SHA-256 verification"):
        downloader.download_checkpoint(
            target,
            opener=lambda *_args, **_kwargs: FakeResponse(content),
            expected_sha256="0" * 64,
            retries=1,
        )

    assert not target.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_invalid_existing_checkpoint_is_preserved_and_refused(tmp_path):
    target = tmp_path / "weights.bin"
    target.write_bytes(b"bad existing file")

    with pytest.raises(ValueError, match="Existing DDColor checkpoint"):
        downloader.download_checkpoint(target, expected_sha256="0" * 64)

    assert target.read_bytes() == b"bad existing file"
