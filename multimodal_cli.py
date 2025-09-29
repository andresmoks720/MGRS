#!/usr/bin/env python3
"""Estonian multimodal CLI with OCR→MGRS conversion and SALUTE dialogue."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from typing import Any

import requests
from openai import OpenAI

from config import AppConfig, load_config
from logging_utils import configure_logging, get_logger
from ocr_shared import run_ocr, warn_if_missing_tesseract
from salute import SaluteConversation

def speak(text: str) -> None:
    print(f"[SPEAK] {text}")


def record_and_transcribe(sec: int = 5) -> str:  # noqa: ARG001 - placeholder shim
    return input("[STT] (type response) ")


class GmtTimeError(RuntimeError):
    """Raised when the GMT API returns malformed data."""


def fetch_gmt_time() -> str:
    """Return the current GMT timestamp using worldtimeapi.org."""

    resp = requests.get("https://worldtimeapi.org/api/timezone/Etc/GMT", timeout=10)
    resp.raise_for_status()
    try:
        payload: dict[str, Any] = resp.json()
    except ValueError as exc:  # json() failed to decode response
        raise GmtTimeError("Invalid JSON payload from GMT API") from exc

    dt_raw = payload.get("datetime")
    if not isinstance(dt_raw, str):
        raise GmtTimeError("GMT API response missing datetime string")

    try:
        dt = datetime.fromisoformat(dt_raw)
    except ValueError as exc:
        raise GmtTimeError("GMT API datetime string was malformed") from exc

    return dt.strftime("%Y-%m-%d %H:%M:%S %Z%z")


def salute_dialogue(conversation: SaluteConversation) -> None:
    """Drive the SALUTE prompts using stdin-backed responses."""

    while True:
        print(f"[SALUTE_PROMPT] {conversation.prompt()}")
        response = record_and_transcribe()
        result = conversation.handle(response)
        if result.completed:
            print("[SALUTE_REPORT]")
            for line in result.report_lines or ():
                print(line)
            break


def build_client(config: AppConfig) -> OpenAI:
    return OpenAI(api_key=config.openai_api_key)


def run_cli(args, config: AppConfig) -> None:
    logger = get_logger("multimodal_cli")
    missing_tesseract = warn_if_missing_tesseract(logger)

    client: OpenAI | None = None
    if config.openai_api_key:
        client = build_client(config)
    else:
        if missing_tesseract:
            sys.exit(
                "❌  Tesseract puudub ja GPT varuplaan on keelatud (OpenAI võti puudub)."
            )
        logger.info("GPT vision fallback disabled: OpenAI API key not configured.")

    def report_raw(engine: str, text: str) -> None:
        print(f"\n=== {engine} ===\n", text or "[tühi]")

    try:
        ocr_result = run_ocr(
            args.image,
            client,
            logger=logger,
            on_raw=report_raw,
            geo=config.geo,
        )
    except Exception as exc:  # pragma: no cover - network failure path
        logger.error("OCR pipeline failed: %s", exc)
        ocr_result = None

    if not ocr_result:
        sys.exit("❌  Koordinaate ei saanud kätte.")

    print(f"[INFO] Kasutasin {ocr_result.engine} OCR-i (✔︎)")
    print(f"\n[RESULT] ({ocr_result.engine}) Koordinaadid leitud. {ocr_result.decimal}")
    print(f"[MGRS] {ocr_result.mgrs}")

    speak(f"Koordinaadid leitud. {ocr_result.decimal}")

    if args.show_gmt:
        try:
            print(f"[GMT] {fetch_gmt_time()}")
        except (requests.RequestException, GmtTimeError) as exc:
            logger.warning("GMT fetch failed: %s", exc)

    if not args.no_salute:
        conversation = SaluteConversation(
            fields=config.salute_fields,
            location=ocr_result.mgrs,
        )
        salute_dialogue(conversation)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estonian multimodal CLI")
    parser.add_argument("--image", required=True, help="Pilt GPS-koordinaatidega")
    parser.add_argument(
        "--no-salute",
        action="store_true",
        help="Ära käivita SALUTE dialoogi",
    )
    parser.add_argument(
        "--show-gmt",
        action="store_true",
        help="Kuva praegune GMT kellaaeg worldtimeapi.org teenusest",
    )
    args = parser.parse_args()

    configure_logging()

    try:
        config = load_config(require_openai=False)
    except RuntimeError as exc:
        sys.exit(f"❌  {exc}")

    run_cli(args, config)


if __name__ == "__main__":
    main()
