"""Application configuration helpers and shared constants."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class GeoBounds:
    """Geographic constraints for validating OCR results."""

    lat_min: float = 55.0
    lat_max: float = 60.0
    lon_min: float = 21.0
    lon_max: float = 28.0
    ref_lat: float = 58.0
    ref_lon: float = 25.0
    radius_km: float = 500.0


@dataclass(frozen=True)
class SaluteField:
    """Definition of a SALUTE field and its Estonian label."""

    key: str
    label: str


DEFAULT_GEO_BOUNDS = GeoBounds()
DEFAULT_SALUTE_FIELDS: Tuple[SaluteField, ...] = (
    SaluteField("size", "Suurus"),
    SaluteField("activity", "Tegevus"),
    SaluteField("location", "Asukoht"),
    SaluteField("unit", "Üksus"),
    SaluteField("time", "Aeg"),
    SaluteField("equipment", "Varustus"),
)


@dataclass(frozen=True)
class AppConfig:
    """Loaded application settings with optional tokens."""

    openai_api_key: str
    telegram_api_token: Optional[str]
    geo: GeoBounds = DEFAULT_GEO_BOUNDS
    salute_fields: Tuple[SaluteField, ...] = DEFAULT_SALUTE_FIELDS


def load_config(*, require_openai: bool = False, require_telegram: bool = False) -> AppConfig:
    """Load configuration from environment variables."""

    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    telegram_api_token = os.getenv("TELEGRAM_API_TOKEN")

    if require_openai and not openai_api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment.")
    if require_telegram and not telegram_api_token:
        raise RuntimeError("Set TELEGRAM_API_TOKEN in your environment.")

    return AppConfig(openai_api_key=openai_api_key, telegram_api_token=telegram_api_token)


__all__ = [
    "AppConfig",
    "GeoBounds",
    "SaluteField",
    "DEFAULT_GEO_BOUNDS",
    "DEFAULT_SALUTE_FIELDS",
    "load_config",
]
