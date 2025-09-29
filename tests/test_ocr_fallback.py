"""Tests for OCR fallback behaviour when Tesseract is unavailable."""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from types import SimpleNamespace

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


def _make_dummy_client():
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda *args, **kwargs: None  # pragma: no cover - stubbed by retry
            )
        )
    )


def test_gpt_vision_handles_missing_choices(
    monkeypatch: pytest.MonkeyPatch, sample_image: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """gpt_vision_ocr should warn and return empty text when choices are missing."""

    class DummyResponse:
        choices: list = []

    def fake_retry(*args, **kwargs):  # type: ignore[no-untyped-def]
        return DummyResponse()

    monkeypatch.setattr(ocr_shared, "retry", fake_retry)

    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("test_missing_choices")

    result = ocr_shared.gpt_vision_ocr(
        str(sample_image), client=_make_dummy_client(), logger=logger
    )

    assert result == ""
    assert "missing completion choices" in caplog.text


def test_gpt_vision_flattens_list_content(
    monkeypatch: pytest.MonkeyPatch, sample_image: Path
) -> None:
    """Ensure list-based message content is flattened into a string."""

    class DummyMessage:
        def __init__(self, content):
            self.content = content

    class DummyChoice:
        def __init__(self, content):
            self.message = DummyMessage(content)

    class DummyResponse:
        def __init__(self, content):
            self.choices = [DummyChoice(content)]

    payload = [
        {"type": "text", "text": "59.437000° N"},
        {"type": "text", "text": "24.753600° E"},
    ]

    def fake_retry(*args, **kwargs):  # type: ignore[no-untyped-def]
        return DummyResponse(payload)

    monkeypatch.setattr(ocr_shared, "retry", fake_retry)

    result = ocr_shared.gpt_vision_ocr(
        str(sample_image),
        client=_make_dummy_client(),
        logger=logging.getLogger("test_flatten"),
    )

    assert result == "59.437000° N\n24.753600° E"


def test_run_ocr_warns_when_fallback_unavailable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, sample_image: Path
) -> None:
    """run_ocr should log a warning and return None if GPT fallback is disabled."""

    monkeypatch.setattr(ocr_shared, "tesseract_ocr", lambda *_, **__: "")

    called = False

    def fail_gpt(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        return ""

    monkeypatch.setattr(ocr_shared, "gpt_vision_ocr", fail_gpt)

    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("test_no_fallback")

    result = ocr_shared.run_ocr(str(sample_image), client=None, logger=logger)

    assert result is None
    assert called is False
    assert "fallback unavailable" in caplog.text
