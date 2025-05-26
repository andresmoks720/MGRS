#!/usr/bin/env python3
"""
telegram_bot.py - May 2025

Telegram bot that receives an image, extracts GPS coordinates via Tesseract OCR
(with GPT-vision fallback), converts them to MGRS, and replies to the user.
Includes verbose debug logging.
"""

# Standard library
import os
import random
import re
import time
import math
import shutil
import logging
from mimetypes import guess_type
import base64

# Telegram imports (python-telegram-bot >=20)
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Third-party
from openai import OpenAI
import pytesseract
from PIL import Image, ImageEnhance
from mgrs import MGRS

# API tokens
TELEGRAM_API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not TELEGRAM_API_TOKEN:
    raise SystemExit("❌  Set TELEGRAM_API_TOKEN in your environment.")
if not OPENAI_API_KEY:
    raise SystemExit("❌  Set OPENAI_API_KEY in your environment.")
client = OpenAI(api_key=OPENAI_API_KEY)

# Geographic constants
LAT_MIN, LAT_MAX = 55.0, 60.0
LON_MIN, LON_MAX = 21.0, 28.0
REF_LAT, REF_LON = 58.0, 25.0
RADIUS_KM = 500.0  # sanity-check radius around reference point

# Logging setup
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")
if shutil.which("tesseract") is None:
    logging.warning("Tesseract not in PATH – OCR fallback will fail")


# Utility: retry with exponential backoff


def retry(fn, *args, retries: int = 3, **kwargs):
    for i in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # pragma: no cover - network calls
            if i == retries:
                raise
            backoff = 2**i + random.random()
            logging.debug("[Retry %d/%d] %s -> %.1fs", i, retries, e, backoff)
            time.sleep(backoff)


# Helper functions


def to_data_url(path: str) -> str:
    mime, _ = guess_type(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime or 'application/octet-stream'};base64,{b64}"


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(h))


# OCR functions


def preprocess(img: Image.Image) -> Image.Image:
    return ImageEnhance.Contrast(img.convert("L")).enhance(2.0)


def tesseract_ocr(path: str) -> str:
    try:
        base = Image.open(path)
    except Exception as e:  # pragma: no cover - corrupted files
        logging.error("Cannot open image %s: %s", path, e)
        return ""

    out = ""
    for angle in (0, 90, 180, 270):
        txt = pytesseract.image_to_string(
            preprocess(base.rotate(angle, expand=True)),
            config="--psm 6 -c tessedit_char_whitelist=0123456789.,NSEW ",
        ).strip()
        logging.debug("[Tesseract %d°] %s", angle, txt or "[no text]")
        out = txt or out
        if ("N" in txt or "S" in txt) and ("E" in txt or "W" in txt):
            if parse_coords(txt):
                break
    return out


def gpt_vision_ocr(path: str) -> str:
    resp = retry(
        client.chat.completions.create,
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Leia drooni ekraanil kuvatud GPS-koordinaadid ja vasta ainult "
                            "kümnend-kraadides kujul, näiteks: 57.927336° N, 26.747699° E"
                        ),
                    },
                    {"type": "image_url", "image_url": to_data_url(path)},
                ],
            }
        ],
        timeout=60,
    )
    return resp.choices[0].message.content.strip()


# Coordinate parsing and MGRS


def parse_coords(txt: str):
    if not txt:
        return None
    t = txt.upper().replace("°", "")
    lat = lon = None
    m1 = re.search(r"(-?\d+\.\d+)\s*([NS])", t)
    m2 = re.search(r"(-?\d+\.\d+)\s*([EW])", t)
    if m1 and m2:
        lat = float(m1.group(1)) * (1 if m1.group(2) == "N" else -1)
        lon = float(m2.group(1)) * (1 if m2.group(2) == "E" else -1)
    else:
        parts = re.split(r"[,\s]+", re.sub(r"[NSEW]", "", t).strip())
        if len(parts) >= 2:
            try:
                lat, lon = float(parts[0]), float(parts[1])
            except ValueError:
                return None
    if lat is None or lon is None:
        return None

    # swap if clearly flipped
    if not (LAT_MIN <= lat <= LAT_MAX) and (LON_MIN <= lon <= LON_MAX):
        lat, lon = lon, lat
    if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
        return None
    if _haversine(lat, lon, REF_LAT, REF_LON) > RADIUS_KM:
        logging.warning(">500 km from reference point: %.6f, %.6f", lat, lon)
    return lat, lon


def to_mgrs(lat: float, lon: float) -> str:
    return MGRS().toMGRS(lat, lon)


def format_mgrs(m: str) -> str:
    z = m[:5]
    r = m[5:]
    h = len(r) // 2
    return f"{z} {r[:h]} {r[h:]}"


# Stub TTS


def speak(text: str):
    logging.info("[SPEAK] %s", text)


# Telegram handlers


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Tere! Saada mulle pilt GPS-koordinaatidega, et ma saaks need välja lugeda."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.debug("[DEBUG] handle_photo for user %s", update.effective_user.id)
    photo = update.message.photo[-1]
    tg_file = await photo.get_file()
    logging.debug("[DEBUG] file_id %s", photo.file_id)

    try:
        path_obj = await tg_file.download_to_drive()
        img_path = str(path_obj)
        logging.debug("[DEBUG] Downloaded to %s", img_path)
    except Exception as e:  # pragma: no cover - network
        logging.error("Download failed: %s", e)
        await update.message.reply_text(f"❌ Faili allalaadimine ebaõnnestus: {e}")
        return

    logging.debug("[DEBUG] Tesseract OCR -> %s", img_path)
    raw = tesseract_ocr(img_path)
    logging.debug("[DEBUG] Tesseract output: %s", raw)
    coords = parse_coords(raw)
    logging.debug("[DEBUG] parse_coords(Tesseract): %s", coords)

    if not coords:
        await update.message.reply_text(f"ℹ️ Tesseract OCR tekst: {raw or '[tühi]'}")
        try:
            logging.debug("[DEBUG] Fallback to GPT-vision OCR")
            raw_llm = gpt_vision_ocr(img_path)
            logging.debug("[DEBUG] GPT output: %s", raw_llm)
            coords = parse_coords(raw_llm)
            logging.debug("[DEBUG] parse_coords(GPT): %s", coords)
        except Exception as e:  # pragma: no cover - network
            logging.error("GPT-vision failed: %s", e)
            await update.message.reply_text(f"❌ LLM-OCR ebaõnnestus: {e}")
            return
        if not coords:
            await update.message.reply_text(
                f"❌ Koordinaate ei saanud kätte. LLM output: {raw_llm or '[tühi]'}"
            )
            return

    lat, lon = coords
    logging.debug("[DEBUG] Final coords: %s", coords)
    dec = f"{abs(lat):.6f}° {'N' if lat>=0 else 'S'}, {abs(lon):.6f}° {'E' if lon>=0 else 'W'}"
    mgrs = format_mgrs(to_mgrs(lat, lon))
    logging.debug("[DEBUG] MGRS result: %s", mgrs)
    await update.message.reply_text(f"📍 Koordinaadid: {dec}\n🗺 MGRS: {mgrs}")
    speak(f"Leidsin koordinaadid: {dec}")


# Main


def main():
    app = ApplicationBuilder().token(TELEGRAM_API_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()


if __name__ == "__main__":
    main()
