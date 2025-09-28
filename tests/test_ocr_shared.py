import pytesseract
from PIL import Image

from ocr_shared import run_ocr


def test_run_ocr_falls_back_to_gpt_when_tesseract_missing(tmp_path, monkeypatch):
    image_path = tmp_path / "dummy.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)

    def raise_not_found(*args, **kwargs):
        raise pytesseract.pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_string", raise_not_found)

    called = {}

    def fake_gpt(path):
        called["path"] = path
        return "gpt result"

    result = run_ocr(image_path, gpt_ocr_fn=fake_gpt)

    assert result == "gpt result"
    assert called["path"] == image_path
