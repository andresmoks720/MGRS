"""Shared OCR and coordinate utilities for the multimodal CLI."""

from __future__ import annotations

import math
import re
from typing import Optional, Tuple

LAT_MIN, LAT_MAX = 55.0, 60.0
LON_MIN, LON_MAX = 21.0, 28.0
REF_LAT, REF_LON = 58.0, 25.0
RADIUS_KM = 500.0


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the distance in kilometres between two decimal degree pairs."""

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


def parse_coords(txt: str) -> Optional[Tuple[float, float]]:
    """Parse decimal latitude/longitude text and validate the result."""

    if not txt:
        return None

    cleaned = txt.upper().replace("°", "")
    lat = lon = None

    match_lat = re.search(r"(-?\d+\.\d+)\s*([NS])", cleaned)
    match_lon = re.search(r"(-?\d+\.\d+)\s*([EW])", cleaned)

    if match_lat and match_lon:
        lat = float(match_lat.group(1)) * (1 if match_lat.group(2) == "N" else -1)
        lon = float(match_lon.group(1)) * (1 if match_lon.group(2) == "E" else -1)
    else:
        parts = re.split(r"[,\s]+", re.sub(r"[NSEW]", "", cleaned).strip())
        if len(parts) >= 2:
            try:
                lat, lon = float(parts[0]), float(parts[1])
            except ValueError:
                return None

    if lat is None or lon is None:
        return None

    if not (LAT_MIN <= lat <= LAT_MAX) and (LAT_MIN <= lon <= LAT_MAX):
        lat, lon = lon, lat

    if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
        return None

    if _haversine(lat, lon, REF_LAT, REF_LON) > RADIUS_KM:
        print("⚠️  >500 km kaugusel kontrollpunktist.")

    return lat, lon


__all__ = [
    "LAT_MIN",
    "LAT_MAX",
    "LON_MIN",
    "LON_MAX",
    "REF_LAT",
    "REF_LON",
    "RADIUS_KM",
    "_haversine",
    "parse_coords",
]
