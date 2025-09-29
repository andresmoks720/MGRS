"""Tests for CLI helpers including Tesseract warnings and GMT fetching."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import multimodal_cli
from config import AppConfig
from ocr_shared import OcrResult


class DummyResponse:
    """Lightweight requests.Response stand-in for testing."""

    def __init__(self, payload=None, *, json_exc: Exception | None = None):
        self._payload = payload
        self._json_exc = json_exc

    def raise_for_status(self) -> None:
        return None

    def json(self):  # type: ignore[no-untyped-def]
        if self._json_exc:
            raise self._json_exc
        return self._payload


def test_warn_if_missing_tesseract_mentions_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Warn about missing Tesseract but reassure users about the GPT fallback."""

    monkeypatch.setattr("ocr_shared.shutil.which", lambda _: None)

    logger = logging.getLogger("multimodal_cli.test")
    with caplog.at_level(logging.WARNING):
        multimodal_cli.warn_if_missing_tesseract(logger)

    assert "GPT vision fallback" in caplog.text


@pytest.mark.parametrize(
    "response",
    [
        DummyResponse(payload={}),
        DummyResponse(payload={"datetime": 123}),
        DummyResponse(payload={"datetime": "invalid"}),
        DummyResponse(json_exc=ValueError("bad json")),
    ],
)
def test_fetch_gmt_time_rejects_malformed_payload(
    monkeypatch: pytest.MonkeyPatch, response: DummyResponse
) -> None:
    """fetch_gmt_time should raise GmtTimeError when payload is malformed."""

    monkeypatch.setattr(multimodal_cli.requests, "get", lambda *_, **__: response)

    with pytest.raises(multimodal_cli.GmtTimeError):
        multimodal_cli.fetch_gmt_time()


def test_run_cli_logs_warning_when_gmt_payload_bad(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """run_cli should catch GmtTimeError and log a warning instead of crashing."""

    args = SimpleNamespace(image="dummy.png", show_gmt=True, no_salute=True)
    config = AppConfig(openai_api_key="test", telegram_api_token=None)

    ocr_result = OcrResult(lat=59.0, lon=24.0, engine="GPT", raw_text="")

    monkeypatch.setattr(multimodal_cli, "build_client", lambda *_: object())
    monkeypatch.setattr(multimodal_cli, "run_ocr", lambda *_, **__: ocr_result)
    def bad_fetch() -> str:
        raise multimodal_cli.GmtTimeError("bad payload")

    monkeypatch.setattr(multimodal_cli, "fetch_gmt_time", bad_fetch)
    monkeypatch.setattr(multimodal_cli, "warn_if_missing_tesseract", lambda *_: False)

    with caplog.at_level(logging.WARNING, logger="multimodal_cli"):
        multimodal_cli.run_cli(args, config)

    assert any("GMT fetch failed" in record.getMessage() for record in caplog.records)


def test_run_cli_allows_tesseract_only_mode(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """run_cli should skip GPT setup when no API key is configured."""

    args = SimpleNamespace(image="dummy.png", show_gmt=False, no_salute=True)
    config = AppConfig(openai_api_key="", telegram_api_token=None)

    ocr_result = OcrResult(lat=59.0, lon=24.0, engine="Tesseract", raw_text="")

    def fail_build_client(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("build_client should not be called without API key")

    def fake_run_ocr(path, client, **kwargs):  # type: ignore[no-untyped-def]
        assert client is None
        return ocr_result

    monkeypatch.setattr(multimodal_cli, "build_client", fail_build_client)
    monkeypatch.setattr(multimodal_cli, "run_ocr", fake_run_ocr)
    monkeypatch.setattr(multimodal_cli, "warn_if_missing_tesseract", lambda *_: False)

    with caplog.at_level(logging.INFO, logger="multimodal_cli"):
        multimodal_cli.run_cli(args, config)

    assert "fallback disabled" in caplog.text


def test_run_cli_exits_when_no_tesseract_and_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tesseract-only mode should exit when Tesseract is missing."""

    args = SimpleNamespace(image="dummy.png", show_gmt=False, no_salute=True)
    config = AppConfig(openai_api_key="", telegram_api_token=None)

    monkeypatch.setattr(multimodal_cli, "warn_if_missing_tesseract", lambda *_: True)

    with pytest.raises(SystemExit) as excinfo:
        multimodal_cli.run_cli(args, config)

    assert "OpenAI" in str(excinfo.value)
