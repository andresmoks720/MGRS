"""Shared OCR helpers for CLI and tests."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional, Union, Any

import pytesseract
from PIL import Image


ImagePath = Union[str, Path]
PreprocessFn = Callable[[Image.Image], Image.Image]
GptFallback = Callable[[ImagePath], str]


def _log_warning(logger: Optional[Any], message: str) -> None:
    if logger is not None and hasattr(logger, "warning"):
        logger.warning(message)
    else:
        print(f"⚠️  {message}", file=sys.stderr)


def tesseract_ocr(
    image_path: ImagePath,
    *,
    preprocess: Optional[PreprocessFn] = None,
    logger: Optional[Any] = None,
    **kwargs: Any,
) -> str:
    """Run Tesseract OCR and return the extracted text."""
    with Image.open(image_path) as image:
        processed = preprocess(image) if preprocess else image
        try:
            return pytesseract.image_to_string(processed, **kwargs)
        except pytesseract.pytesseract.TesseractNotFoundError:
            _log_warning(
                logger,
                "Tesseract executable not found; falling back to GPT OCR.",
            )
            return ""


def run_ocr(
    image_path: ImagePath,
    *,
    preprocess: Optional[PreprocessFn] = None,
    logger: Optional[Any] = None,
    gpt_ocr_fn: Optional[GptFallback] = None,
    **kwargs: Any,
) -> str:
    text = tesseract_ocr(
        image_path,
        preprocess=preprocess,
        logger=logger,
        **kwargs,
    ).strip()

    if text:
        return text

    if gpt_ocr_fn is None:
        return ""

    return gpt_ocr_fn(image_path)
