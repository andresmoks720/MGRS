"""Shared OCR utilities for the multimodal CLI.

This module houses helpers that coordinate the two OCR engines used by the
project – the local Tesseract binary and the GPT based fallback.  The
functions here are intentionally thin so they can be unit tested without
network access.  Higher level code is expected to monkeypatch
``gpt_vision_ocr`` during testing to avoid real API calls.
"""

from __future__ import annotations

from dataclasses import dataclass
import enum
import logging
from typing import Optional, Tuple

from PIL import Image
import pytesseract


LOGGER = logging.getLogger(__name__)


class Engine(enum.Enum):
    """Enum describing which OCR engine produced the result."""

    TESSERACT = "tesseract"
    GPT = "gpt"


@dataclass
class OcrResult:
    """Container for OCR output and any parsed coordinates."""

    engine: Engine
    text: str
    coordinates: Optional[Tuple[float, float]]


def parse_coordinates(text: str) -> Optional[Tuple[float, float]]:
    """Parse latitude/longitude pairs from OCR text.

    The parser is intentionally liberal: it accepts either cardinal directions
    (``57.0 N, 25.0 E``) or bare decimal pairs separated by whitespace or a
    comma.  The caller is responsible for performing additional geographic
    validation.
    """

    if not text:
        return None

    clean = text.strip().upper().replace("°", "")
    if not clean:
        return None

    lat = lon = None
    # Coordinate with cardinal directions
    parts = clean.replace("N", " N").replace("S", " S").replace("E", " E").replace("W", " W")
    tokens = [token for token in parts.replace(",", " ").split() if token]
    if len(tokens) >= 4:
        try:
            lat_val = float(tokens[0])
            lat_dir = tokens[1]
            lon_val = float(tokens[2])
            lon_dir = tokens[3]
        except ValueError:
            lat_val = lon_val = None
        else:
            lat = lat_val if lat_dir == "N" else -lat_val
            lon = lon_val if lon_dir == "E" else -lon_val

    if lat is None or lon is None:
        stripped = clean
        for ch in "NSEW":
            stripped = stripped.replace(ch, "")
        parts = [p for p in stripped.replace(",", " ").split() if p]
        if len(parts) >= 2:
            try:
                lat = float(parts[0])
                lon = float(parts[1])
            except ValueError:
                return None

    if lat is None or lon is None:
        return None

    return lat, lon


def tesseract_ocr(image_path: str) -> str:
    """Run Tesseract OCR on the provided image path.

    If the Tesseract binary is unavailable, ``pytesseract`` raises
    ``TesseractNotFoundError``.  The function catches the exception, logs a
    warning, and returns an empty string so the caller can fall back to GPT
    based OCR.
    """

    try:
        image = Image.open(image_path)
    except Exception as exc:  # pragma: no cover - relies on PIL internals
        LOGGER.error("Failed to open image %s: %s", image_path, exc)
        return ""

    try:
        return pytesseract.image_to_string(image).strip()
    except pytesseract.pytesseract.TesseractNotFoundError:
        LOGGER.warning(
            "Tesseract executable not found – falling back to GPT OCR for %s.",
            image_path,
        )
        return ""


def gpt_vision_ocr(image_path: str) -> str:  # pragma: no cover - network call
    """Placeholder GPT vision OCR implementation.

    The real implementation lives in the CLI module where OpenAI credentials
    are handled.  Tests monkeypatch this function.
    """

    raise NotImplementedError("GPT OCR requires network access and is stubbed in tests.")


def run_ocr(image_path: str) -> OcrResult:
    """Run OCR using Tesseract first, falling back to GPT vision if needed."""

    tesseract_text = tesseract_ocr(image_path)
    coordinates = parse_coordinates(tesseract_text)
    if coordinates:
        return OcrResult(engine=Engine.TESSERACT, text=tesseract_text, coordinates=coordinates)

    LOGGER.info("Falling back to GPT OCR for %s", image_path)
    gpt_text = gpt_vision_ocr(image_path)
    gpt_coordinates = parse_coordinates(gpt_text)
    return OcrResult(engine=Engine.GPT, text=gpt_text, coordinates=gpt_coordinates)

