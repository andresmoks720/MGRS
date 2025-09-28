"""Tests for OCR coordinate parsing helpers."""

import pytest

import ocr_shared


def test_parse_coords_swaps_lat_lon_when_needed():
    lat, lon = ocr_shared.parse_coords("26.123 58.456")

    assert lat == pytest.approx(58.456)
    assert lon == pytest.approx(26.123)


def test_parse_coords_rejects_out_of_bounds():
    assert ocr_shared.parse_coords("54.0 N, 27.0 E") is None
    assert ocr_shared.parse_coords("57.0 N, 30.0 E") is None


def test_parse_coords_warns_when_over_500_km(monkeypatch, capsys):
    monkeypatch.setattr(ocr_shared, "_haversine", lambda *args, **kwargs: 600.0)

    result = ocr_shared.parse_coords("58.5 N, 26.0 E")

    captured = capsys.readouterr()

    assert "⚠️" in captured.out
    assert result == (58.5, 26.0)
