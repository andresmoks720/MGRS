"""Telegram bot interface for the multimodal OCR tool.

The bot accepts an image with GPS coordinates, extracts the location using
Tesseract with an OpenAI vision fallback and then guides the user through a
SALUTE dialogue in Estonian. Voice or external TTS are not used; responses are
returned as Telegram text messages.
"""

import os
import tempfile
from typing import Dict, List, Tuple

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)

from multimodal_cli import (
    tesseract_ocr,
    gpt_vision_ocr,
    parse_coords,
    to_mgrs,
    format_mgrs,
)

TELEGRAM_API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_API_TOKEN:
    raise SystemExit("❌  Set TELEGRAM_API_TOKEN in your environment.")
if not OPENAI_API_KEY:
    raise SystemExit("❌  Set OPENAI_API_KEY in your environment.")

FIELDS: List[Tuple[str, str]] = [
    ("size", "Suurus"),
    ("activity", "Tegevus"),
    ("location", "Asukoht"),
    ("unit", "Üksus"),
    ("time", "Aeg"),
    ("equipment", "Varustus"),
]
SALUTE = 1


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
