import logging
import sys
from pathlib import Path

import pytest
from PIL import Image
import pytesseract

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ocr_shared


def create_dummy_image(path: Path) -> None:
    image = Image.new("RGB", (10, 10), color="white")
    image.save(path)


def test_run_ocr_tesseract_missing_triggers_fallback(monkeypatch, tmp_path, caplog):
    image_path = tmp_path / "dummy.png"
    create_dummy_image(image_path)

    def _raise_missing(*args, **kwargs):
        raise pytesseract.pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(ocr_shared.pytesseract, "image_to_string", _raise_missing)
    monkeypatch.setattr(ocr_shared, "gpt_vision_ocr", lambda _path: "58.0 N, 25.0 E")

    caplog.set_level(logging.WARNING)

    result = ocr_shared.run_ocr(str(image_path))

    assert result.engine is ocr_shared.Engine.GPT
    assert result.coordinates == (58.0, 25.0)
    assert "Tesseract executable not found" in caplog.text

