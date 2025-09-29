"""Tests for coordinate parsing helpers used in OCR flows."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pytest

from config import DEFAULT_GEO_BOUNDS
from ocr_shared import MISSING_TESSERACT_WARNING, parse_coords, warn_if_missing_tesseract


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            f"{DEFAULT_GEO_BOUNDS.ref_lat} {DEFAULT_GEO_BOUNDS.ref_lon}",
            (DEFAULT_GEO_BOUNDS.ref_lat, DEFAULT_GEO_BOUNDS.ref_lon),
        ),
        (
            "59,437 N 24,753 E",
            (59.437, 24.753),
        ),
        (
            f"{DEFAULT_GEO_BOUNDS.lon_min + 1.5} {DEFAULT_GEO_BOUNDS.lat_max - 1.0}",
            (DEFAULT_GEO_BOUNDS.lat_max - 1.0, DEFAULT_GEO_BOUNDS.lon_min + 1.5),
        ),
        (
            f"{DEFAULT_GEO_BOUNDS.ref_lat} N {DEFAULT_GEO_BOUNDS.ref_lon} E",
            (DEFAULT_GEO_BOUNDS.ref_lat, DEFAULT_GEO_BOUNDS.ref_lon),
        ),
    ],
)
def test_parse_coords_success_cases(text: str, expected: tuple[float, float]) -> None:
    """parse_coords should accept normal, swapped, and suffixed values."""

    result = parse_coords(text, geo=DEFAULT_GEO_BOUNDS)
    assert result == pytest.approx(expected)  # type: ignore[arg-type]


def test_parse_coords_returns_none_when_out_of_bounds() -> None:
    """Values outside the configured bounds should be rejected."""

    text = f"{DEFAULT_GEO_BOUNDS.lat_max + 1.0} {DEFAULT_GEO_BOUNDS.lon_max + 1.0}"
    assert parse_coords(text, geo=DEFAULT_GEO_BOUNDS) is None


def test_parse_coords_logs_warning_when_far_from_reference() -> None:
    """Coordinates beyond the allowed radius emit a warning."""

    shrunk_bounds = replace(DEFAULT_GEO_BOUNDS, radius_km=100.0)
    logger = Mock()
    text = f"{DEFAULT_GEO_BOUNDS.lat_min} {DEFAULT_GEO_BOUNDS.lon_min}"

    result = parse_coords(text, logger=logger, geo=shrunk_bounds)

    assert result == pytest.approx(
        (DEFAULT_GEO_BOUNDS.lat_min, DEFAULT_GEO_BOUNDS.lon_min)
    )  # type: ignore[arg-type]
    logger.warning.assert_called_once()


def test_warn_if_missing_tesseract_logs_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """warn_if_missing_tesseract should log when the binary is absent."""

    monkeypatch.setattr("ocr_shared.shutil.which", lambda _: None)
    logger = Mock()

    missing = warn_if_missing_tesseract(logger)

    assert missing is True
    logger.warning.assert_called_once_with(MISSING_TESSERACT_WARNING)


def test_warn_if_missing_tesseract_returns_false_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """warn_if_missing_tesseract should return False when the binary exists."""

    monkeypatch.setattr("ocr_shared.shutil.which", lambda _: "/usr/bin/tesseract")
    logger = Mock()

    missing = warn_if_missing_tesseract(logger)

    assert missing is False
    logger.warning.assert_not_called()
