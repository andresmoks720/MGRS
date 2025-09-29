"""Shared OCR and coordinate utilities for the CLI and Telegram workflows."""
from __future__ import annotations

import base64
import logging
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from mimetypes import guess_type
from typing import Callable, Iterable, Optional, Tuple

import pytesseract
from pytesseract.pytesseract import TesseractNotFoundError
from PIL import Image, ImageEnhance
from mgrs import MGRS

from config import DEFAULT_GEO_BOUNDS, GeoBounds

LAT_MIN = DEFAULT_GEO_BOUNDS.lat_min
LAT_MAX = DEFAULT_GEO_BOUNDS.lat_max
LON_MIN = DEFAULT_GEO_BOUNDS.lon_min
LON_MAX = DEFAULT_GEO_BOUNDS.lon_max
REF_LAT = DEFAULT_GEO_BOUNDS.ref_lat
REF_LON = DEFAULT_GEO_BOUNDS.ref_lon
RADIUS_KM = DEFAULT_GEO_BOUNDS.radius_km

_DEFAULT_PROMPT = (
    "Leia drooni ekraanil kuvatud GPS-koordinaadid ja vasta ainult kümnendkraadides, "
    "nt: 57.927336° N, 26.747699° E"
)
_DEFAULT_ANGLES: Tuple[int, ...] = (0, 90, 180, 270)


@dataclass
class OcrResult:
    """Normalized OCR output containing both coordinate formats."""

    lat: float
    lon: float
    engine: str
    raw_text: str

    @property
    def decimal(self) -> str:
        return format_decimal(self.lat, self.lon)

    @property
    def mgrs(self) -> str:
        return format_mgrs(to_mgrs(self.lat, self.lon))


def retry(
    fn: Callable,
    *args,
    retries: int = 3,
    logger: Optional[logging.Logger] = None,
    **kwargs,
):
    """Retry helper with exponential backoff and jitter."""

    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - network failure path
            if attempt == retries:
                raise
            backoff = 2 ** attempt + random.random()
            if logger:
                logger.debug("[retry %d/%d] %s → %.1fs", attempt, retries, exc, backoff)
            else:
                print(
                    f"[Retry {attempt}/{retries}] {fn.__name__}: {exc} → {backoff:.1f}s",
                    file=sys.stderr,
                )
            time.sleep(backoff)


def to_data_url(path: str) -> str:
    """Return the file contents encoded as a data URL."""

    mime, _ = guess_type(path)
    with open(path, "rb") as handle:
        b64 = base64.b64encode(handle.read()).decode()
    return f"data:{mime or 'image/jpeg'};base64,{b64}"


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return radius * 2 * math.asin(math.sqrt(a))


def preprocess(img: Image.Image) -> Image.Image:
    """Greyscale and boost contrast to aid OCR accuracy."""

    return ImageEnhance.Contrast(img.convert("L")).enhance(2.0)


def tesseract_ocr(
    path: str,
    *,
    angles: Iterable[int] = _DEFAULT_ANGLES,
    psm: int = 7,
    whitelist: str = "0123456789.,NSEW° ",
    logger: Optional[logging.Logger] = None,
    geo: GeoBounds = DEFAULT_GEO_BOUNDS,
) -> str:
    """Run Tesseract OCR against the provided image path."""

    try:
        base = Image.open(path)
    except Exception as exc:  # pragma: no cover - file failure path
        if logger:
            logger.error("Cannot open %s: %s", path, exc)
        else:
            print("❌  Cannot open image:", exc, file=sys.stderr)
        return ""

    best_txt = ""
    try:
        for angle in angles:
            try:
                txt = pytesseract.image_to_string(
                    preprocess(base.rotate(angle, expand=True)),
                    config=f"--psm {psm} -c tessedit_char_whitelist={whitelist}",
                ).strip()
            except TesseractNotFoundError:
                message = "Tesseract binary not found; skipping to GPT fallback."
                if logger:
                    logger.warning(message)
                else:
                    print(f"⚠️  {message}", file=sys.stderr)
                return ""
            if logger:
                logger.debug("[Tesseract %d°] %s", angle, txt or "[no text]")
            else:
                print(f"\n[Tesseract {angle}°]\n{txt or '[no text]'}")
            if txt:
                best_txt = txt
            if parse_coords(txt, logger=logger, geo=geo):
                return txt
    except TesseractNotFoundError:
        message = "Tesseract binary not found; skipping to GPT fallback."
        if logger:
            logger.warning(message)
        else:
            print(f"⚠️  {message}", file=sys.stderr)
        return ""
    return best_txt


def gpt_vision_ocr(
    path: str,
    client,
    *,
    prompt: str = _DEFAULT_PROMPT,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Query the GPT-4o-mini vision endpoint for coordinates."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": to_data_url(path)}},
            ],
        }
    ]
    resp = retry(
        client.chat.completions.create,
        model="gpt-4o-mini",
        messages=messages,
        timeout=60,
        logger=logger,
    )
    return resp.choices[0].message.content.strip()


def _strip_dir_suffix(value: str) -> str:
    return re.sub(r"[NSEW]$", "", value, flags=re.IGNORECASE)


def parse_coords(
    txt: Optional[str],
    *,
    logger: Optional[logging.Logger] = None,
    geo: GeoBounds = DEFAULT_GEO_BOUNDS,
) -> Optional[Tuple[float, float]]:
    """Normalize decimal-degree coordinates from OCR text."""

    if not txt:
        return None
    t = txt.upper().replace("°", " ")

    match_lat = re.search(r"(-?\d+\.\d+)\s*([NS])", t)
    match_lon = re.search(r"(-?\d+\.\d+)\s*([EW])", t)
    if match_lat and match_lon:
        lat = float(match_lat.group(1)) * (1 if match_lat.group(2) == "N" else -1)
        lon = float(match_lon.group(1)) * (1 if match_lon.group(2) == "E" else -1)
    else:
        parts = [_strip_dir_suffix(p) for p in re.split(r"[,\s]+", t) if p]
        parts = [p for p in parts if p]
        if len(parts) < 2:
            return None
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except ValueError:
            return None

    if not (geo.lat_min <= lat <= geo.lat_max):
        if (geo.lon_min <= lat <= geo.lon_max) and (
            geo.lat_min <= lon <= geo.lat_max
        ):
            lat, lon = lon, lat

    if not (geo.lat_min <= lat <= geo.lat_max and geo.lon_min <= lon <= geo.lon_max):
        return None

    if _haversine(lat, lon, geo.ref_lat, geo.ref_lon) > geo.radius_km:
        if logger:
            logger.warning(">500km from ref: %.6f, %.6f", lat, lon)
        else:
            print("⚠️  >500 km kaugusel kontrollpunktist.")
    return lat, lon


def to_mgrs(lat: float, lon: float) -> str:
    return MGRS().toMGRS(lat, lon)


def format_mgrs(mgrs: str) -> str:
    zone = mgrs[:5]
    remainder = mgrs[5:]
    half = len(remainder) // 2
    return f"{zone} {remainder[:half]} {remainder[half:]}"


def format_decimal(lat: float, lon: float) -> str:
    return (
        f"{abs(lat):.6f}° {'N' if lat >= 0 else 'S'}, "
        f"{abs(lon):.6f}° {'E' if lon >= 0 else 'W'}"
    )


def run_ocr(
    path: str,
    client,
    *,
    logger: Optional[logging.Logger] = None,
    on_raw: Optional[Callable[[str, str], None]] = None,
    geo: GeoBounds = DEFAULT_GEO_BOUNDS,
) -> Optional[OcrResult]:
    """Execute the OCR pipeline and return structured coordinates if found."""

    raw = tesseract_ocr(path, logger=logger, geo=geo)
    if on_raw:
        on_raw("Tesseract OCR", raw)
    coords = parse_coords(raw, logger=logger, geo=geo)
    if coords:
        lat, lon = coords
        return OcrResult(lat=lat, lon=lon, engine="Tesseract", raw_text=raw)

    raw = gpt_vision_ocr(path, client, logger=logger)
    if on_raw:
        on_raw("GPT-4o-mini Vision", raw)
    coords = parse_coords(raw, logger=logger, geo=geo)
    if not coords:
        return None

    lat, lon = coords
    return OcrResult(lat=lat, lon=lon, engine="GPT-4o-mini Vision", raw_text=raw)


__all__ = [
    "LAT_MIN",
    "LAT_MAX",
    "LON_MIN",
    "LON_MAX",
    "REF_LAT",
    "REF_LON",
    "RADIUS_KM",
    "OcrResult",
    "retry",
    "to_data_url",
    "preprocess",
    "tesseract_ocr",
    "gpt_vision_ocr",
    "parse_coords",
    "to_mgrs",
    "format_mgrs",
    "format_decimal",
    "run_ocr",
]
