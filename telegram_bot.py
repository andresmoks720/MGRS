"""Telegram bot interface for the multimodal OCR tool.

The bot accepts an image with GPS coordinates, extracts the location using
Tesseract with an OpenAI vision fallback and then guides the user through a
SALUTE dialogue in Estonian. Voice or external TTS are not used; responses are
returned as Telegram text messages.
"""

import os
import random
import re
import tempfile
import time
import math
import shutil
import base64
from mimetypes import guess_type
from typing import Dict, List, Tuple

from openai import OpenAI
import pytesseract
from PIL import Image, ImageEnhance
from mgrs import MGRS

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)


TELEGRAM_API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_API_TOKEN:
    raise SystemExit("❌  Set TELEGRAM_API_TOKEN in your environment.")
if not OPENAI_API_KEY:
    raise SystemExit("❌  Set OPENAI_API_KEY in your environment.")

client = OpenAI(api_key=OPENAI_API_KEY)

LAT_MIN, LAT_MAX = 55.0, 60.0
LON_MIN, LON_MAX = 21.0, 28.0
REF_LAT, REF_LON = 58.0, 25.0
RADIUS_KM = 500.0

if shutil.which("tesseract") is None:
    print("⚠️  Tesseract not in PATH – OCR fallback will fail")

FIELDS: List[Tuple[str, str]] = [
    ("size", "Suurus"),
    ("activity", "Tegevus"),
    ("location", "Asukoht"),
    ("unit", "Üksus"),
    ("time", "Aeg"),
    ("equipment", "Varustus"),
]
SALUTE = 1


def retry(fn, *args, retries: int = 3, **kwargs):
    for i in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if i == retries:
                raise
            backoff = 2**i + random.random()
            print(f"[Retry {i}/{retries}] {fn.__name__}: {e} → {backoff:.1f}s")
            time.sleep(backoff)


def to_data_url(path: str) -> str:
    mime, _ = guess_type(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime or 'application/octet-stream'};base64,{b64}"


def _haversine(a1: float, a2: float, b1: float, b2: float) -> float:
    R = 6371
    dlat = math.radians(b1 - a1)
    dlon = math.radians(b2 - a2)
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(a1))
        * math.cos(math.radians(b1))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(h))


def preprocess(img):
    return ImageEnhance.Contrast(img.convert("L")).enhance(2.0)


def tesseract_ocr(path: str) -> str:
    try:
        base = Image.open(path)
    except Exception as e:
        print(f"❌  Cannot open image: {e}")
        return ""
    out = ""
    for angle in (0, 90, 180, 270):
        txt = pytesseract.image_to_string(
            preprocess(base.rotate(angle, expand=True)),
            config="--psm 6 -c tessedit_char_whitelist=0123456789.,NSEW ",
        ).strip()
        print(f"[Tesseract {angle}°] {txt or '[no text]'}")
        out = txt or out
        if ("N" in txt or "S" in txt) and ("E" in txt or "W" in txt):
            if parse_coords(txt):
                break
    return out


def gpt_vision_ocr(path: str) -> str:
    resp = retry(
        client.chat.completions.create,
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Leia drooni ekraanil kuvatud GPS-koordinaadid ja vasta ainult "
                            "kümnend-kraadides kujul, näiteks: 57.927336° N, 26.747699° E"
                        ),
                    },
                    {"type": "input_image", "image_url": to_data_url(path)},
                ],
            }
        ],
        timeout=60,
    )
    return resp.output_text.strip()


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
    if not (LAT_MIN <= lat <= LAT_MAX) and (LON_MIN <= lon <= LON_MAX):
        lat, lon = lon, lat
    if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
        return None
    if _haversine(lat, lon, REF_LAT, REF_LON) > RADIUS_KM:
        print("⚠️  >500 km kaugusel kontrollpunktist.")
    return lat, lon


def to_mgrs(lat: float, lon: float) -> str:
    return MGRS().toMGRS(lat, lon)


def format_mgrs(m: str) -> str:
    z = m[:5]
    r = m[5:]
    h = len(r) // 2
    return f"{z} {r[:h]} {r[h:]}"


def ocr_pipeline(path: str) -> Tuple[str, str] | Tuple[None, None]:
    """Return decimal coordinates and formatted MGRS for an image."""
    raw = tesseract_ocr(path)
    coords = parse_coords(raw)
    if not coords:
        try:
            raw = gpt_vision_ocr(path)
            coords = parse_coords(raw)
        except Exception as e:  # pragma: no cover - network calls
            print(f"❌  LLM error: {e}")
            return None, None
    if not coords:
        return None, None
    lat, lon = coords
    dec = f"{abs(lat):.6f}° {'N' if lat >= 0 else 'S'}, {abs(lon):.6f}° {'E' if lon >= 0 else 'W'}"
    mgrs = format_mgrs(to_mgrs(lat, lon))
    return dec, mgrs


def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Notify user that the bot is ready."""
    update.message.reply_text("Saada pilt GPS-koordinaatidega.")


def ask_salute(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Prompt the user for the next SALUTE field."""
    idx = context.user_data.get("salute_idx", 0)
    key, et = FIELDS[idx]
    update.message.reply_text(f"Palun ütle {et}. (või 'vahele'/'tagasi')")
    return SALUTE


def salute_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Collect SALUTE answers from the user."""
    txt = update.message.text or ""
    idx = context.user_data.get("salute_idx", 0)
    key, _ = FIELDS[idx]
    ans: Dict[str, str] = context.user_data.setdefault(
        "salute_ans", {"location": context.user_data.get("mgrs", "")}
    )
    low = txt.lower()
    if low.startswith("tagasi"):
        parts = low.split()
        if len(parts) >= 2:
            tgt = parts[1]
            for j, (k, et) in enumerate(FIELDS):
                if tgt == k or tgt == et.lower():
                    context.user_data["salute_idx"] = j
                    return ask_salute(update, context)
    if low in ("vahele", "skip", "järgmine"):
        ans[key] = ""
    else:
        ans[key] = txt
    idx += 1
    context.user_data["salute_idx"] = idx
    if idx >= len(FIELDS):
        lines = [f"{et}: {ans.get(k, '–') or '–'}" for k, et in FIELDS]
        update.message.reply_text("SALUTE raport:\n" + "\n".join(lines))
        return ConversationHandler.END
    return ask_salute(update, context)


def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle incoming photos, perform OCR and start SALUTE."""
    file = update.message.photo[-1]
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        file.get_file().download_to_drive(tmp.name)
        dec, mgrs = ocr_pipeline(tmp.name)
    if not mgrs:
        update.message.reply_text("Koordinaate ei saanud kätte.")
        return ConversationHandler.END
    context.user_data["mgrs"] = mgrs
    msg = f"Koordinaadid leitud. {dec}\nMGRS: {mgrs}"
    update.message.reply_text(msg)
    context.user_data["salute_idx"] = 0
    context.user_data["salute_ans"] = {"location": mgrs}
    return ask_salute(update, context)


def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """End the conversation."""
    update.message.reply_text("Vestlus katkestatud.")
    return ConversationHandler.END


def main() -> None:
    """Run the Telegram bot."""
    app = ApplicationBuilder().token(TELEGRAM_API_TOKEN).build()
    conv = ConversationHandler(
        entry_points=[MessageHandler(filters.PHOTO, photo)],
        states={
            SALUTE: [MessageHandler(filters.TEXT & ~filters.COMMAND, salute_handler)]
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv)
    app.run_polling()


if __name__ == "__main__":
    main()
