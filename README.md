# MGRS OCR Toolkit

Tools for extracting GPS coordinates from imagery, validating them against an
Estonian reference window, converting to MGRS, and gathering structured SALUTE
reports in Estonian.

## Components

### `multimodal_cli.py`
Interactive command-line workflow that:

1. Attempts OCR with Tesseract (rotating through four angles) and falls back to
   the OpenAI GPT-4o-mini vision endpoint when needed.
2. Validates coordinates within 500 km of the Estonia reference point before
   converting them to MGRS.
3. Prompts the user through a SALUTE report flow (stdin stub instead of audio).

Usage:

```bash
export OPENAI_API_KEY=...  # required
python multimodal_cli.py --image path/to/photo.jpg
python multimodal_cli.py --image path/to/photo.jpg --no-salute  # skip dialogue
python multimodal_cli.py --image path/to/photo.jpg --show-gmt    # print GMT clock
```

### `telegram_bot.py`
A synchronous Telegram bot (python-telegram-bot v13) that listens for photos,
extracts coordinates using the same OCR pipeline, replies with decimal + MGRS
coordinates, and continues into the SALUTE questionnaire.

Usage:

```bash
export OPENAI_API_KEY=...
export TELEGRAM_API_TOKEN=...
python telegram_bot.py
```

The bot stores no data server-side. Each chat session keeps the parsed MGRS in
memory only for the SALUTE flow.

### `ocr_shared.py`
Shared OCR, coordinate parsing, and formatting helpers imported by both the CLI
and Telegram entry points. The module owns the Estonia bounds, retry logic, and
an :class:`OcrResult` data class that surfaces the decimal and MGRS strings so
callers don’t have to duplicate parsing or formatting steps.

### `config.py`
Centralises environment-derived configuration (API tokens), Estonia geography
bounds, and the ordered SALUTE fields so both front-ends share one source of
truth.

### `salute.py`
Implements a reusable SALUTE conversation engine that handles prompts,
skipping/backtracking, and report formatting. The CLI and Telegram bot both
instantiate the same helper to keep behaviour aligned.

### `logging_utils.py`
Provides a consistent logging configuration and named loggers for the CLI and
bot so diagnostics and retry messages use a single format.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

`pytesseract` expects the Tesseract binary to be installed and discoverable in
`PATH`.
