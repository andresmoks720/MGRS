"""Tests for OCR fallback behaviour when Tesseract is unavailable."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image
from pytesseract.pytesseract import TesseractNotFoundError

sys.path.append(str(Path(__file__).resolve().parents[1]))

import ocr_shared


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "sample.png"
    Image.new("RGB", (10, 10), color="white").save(path)
    return path


def test_run_ocr_falls_back_when_tesseract_missing(monkeypatch: pytest.MonkeyPatch, sample_image: Path) -> None:
    """run_ocr should return GPT results if Tesseract is not installed."""

    def fake_image_to_string(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise TesseractNotFoundError()

    fallback_text = "59.437000° N, 24.753600° E"

    def fake_gpt_vision_ocr(path: str, client, **kwargs):  # type: ignore[no-untyped-def]
        return fallback_text

    monkeypatch.setattr(ocr_shared.pytesseract, "image_to_string", fake_image_to_string)
    monkeypatch.setattr(ocr_shared, "gpt_vision_ocr", fake_gpt_vision_ocr)

    result = ocr_shared.run_ocr(str(sample_image), client=object())

    assert result is not None
    assert result.engine == "GPT-4o-mini Vision"
    assert result.raw_text == fallback_text
    assert pytest.approx(result.lat, rel=1e-6) == 59.437
    assert pytest.approx(result.lon, rel=1e-6) == 24.7536
