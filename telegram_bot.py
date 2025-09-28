#!/usr/bin/env python3
"""Telegram bot that performs OCR→MGRS conversion and SALUTE collection."""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Optional

from openai import OpenAI
from telegram import Update
from telegram.ext import (
    CallbackContext,
    CommandHandler,
    ConversationHandler,
    Filters,
    MessageHandler,
    Updater,
)

from config import AppConfig, load_config
from logging_utils import configure_logging, get_logger
from ocr_shared import run_ocr
from salute import SaluteConversation

SALUTE = 1

LOGGER = get_logger("telegram_bot")
CONFIG: Optional[AppConfig] = None
CLIENT: Optional[OpenAI] = None


def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(
        "Tere! Saada mulle pilt GPS-koordinaatidega, et ma saaks need välja lugeda."
    )


def get_conversation(context: CallbackContext) -> SaluteConversation:
    config = CONFIG
    if config is None:
        raise RuntimeError("Configuration not initialised")
    conversation = context.user_data.get("salute_conv")
    if isinstance(conversation, SaluteConversation):
        return conversation
    conversation = SaluteConversation(fields=config.salute_fields)
    context.user_data["salute_conv"] = conversation
    return conversation


def ask_salute(update: Update, context: CallbackContext) -> int:
    conversation = get_conversation(context)
    update.message.reply_text(conversation.prompt())
    return SALUTE


def salute_handler(update: Update, context: CallbackContext) -> int:
    conversation = get_conversation(context)
    result = conversation.handle(update.message.text or "")

    if result.completed:
        report = "\n".join(result.report_lines or ())
        update.message.reply_text("SALUTE raport:\n" + report)
        return ConversationHandler.END

    update.message.reply_text(result.prompt or conversation.prompt())
    return SALUTE


def handle_photo(update: Update, context: CallbackContext) -> int:
    assert CLIENT is not None
    assert CONFIG is not None

    photo = update.message.photo[-1]
    tg_file = context.bot.get_file(photo.file_id)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    path = tmp.name

    try:
        tg_file.download(path)
        try:
            result = run_ocr(path, CLIENT, logger=LOGGER, geo=CONFIG.geo)
        except Exception as exc:  # pragma: no cover - network failure path
            LOGGER.error("OCR pipeline failed: %s", exc)
            result = None
    finally:
        try:
            os.unlink(path)
        except OSError:
            LOGGER.debug("Failed to delete temporary file: %s", path)

    if not result:
        update.message.reply_text("Koordinaate ei saanud kätte.")
        return ConversationHandler.END

    update.message.reply_text(f"📍 Koordinaadid: {result.decimal}\n🗺 MGRS: {result.mgrs}")
    context.user_data["salute_conv"] = SaluteConversation(
        fields=CONFIG.salute_fields,
        location=result.mgrs,
    )
    return ask_salute(update, context)


def cancel(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("Vestlus katkestatud.")
    return ConversationHandler.END


def main() -> None:
    configure_logging()

    try:
        config = load_config(require_openai=True, require_telegram=True)
    except RuntimeError as exc:
        raise SystemExit(f"❌  {exc}") from exc

    global CONFIG, CLIENT
    CONFIG = config
    CLIENT = OpenAI(api_key=config.openai_api_key)

    if shutil.which("tesseract") is None:
        LOGGER.warning("Tesseract not in PATH – OCR fallback will fail")

    updater = Updater(token=config.telegram_api_token, use_context=True)
    dispatcher = updater.dispatcher

    conv = ConversationHandler(
        entry_points=[MessageHandler(Filters.photo, handle_photo)],
        states={SALUTE: [MessageHandler(Filters.text & ~Filters.command, salute_handler)]},
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(conv)

    LOGGER.info("Bot starting…")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
