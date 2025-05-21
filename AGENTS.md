# AGENTS.md

## 1. Project Overview
This agent assists with:  
- **Image→GPS OCR** using Tesseract (fallback: GPT-4o-mini vision)  
- **Coordinate validation** within Estonia ± 500 km and MGRS conversion  
- **Speech→Text** via Whisper (stdin simulation) and structured SALUTE prompts  
- **Printed “TTS”** only (no external speech engines)

## 2. Environment & Setup
- **Repository**: `<your-repo-url>` on branch `main`  
- **Python**: 3.10+ (use `pyenv` or system)  
- **Virtualenv**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **Dependencies**:
  ```bash
  pip install -r requirements.txt
  ```
- **Environment**: copy `.env.sample`→`.env`, set:
  ```env
  OPENAI_API_KEY=<your-key>
  ```

## 3. Dependencies & Tools
- **Core libs**: `openai`, `requests`, `pytesseract`, `Pillow`, `mgrs`  
- **CLI**: built with `argparse` (no external CLI frameworks)  
- **OCR**: requires Tesseract installed and in `$PATH`  
- **Fallback LLM OCR**: uses OpenAI’s `gpt-4o-mini` vision model  
- **STT**: simulated via `input()` (replace with Whisper and sounddevice if needed)

## 4. Testing & Quality
- **Unit tests**: use `pytest` (place tests under `tests/`)  
  ```bash
  pytest --maxfail=1 --disable-warnings -q
  ```
- **Linters/Formatters**:  
  ```bash
  flake8 .
  black --line-length 88 .
  ```
- **Type checking** (optional):  
  ```bash
  mypy multimodal_cli.py
  ```

## 5. Coding Conventions
- **Naming**: `snake_case` for functions and variables  
- **Constants**: UPPER_SNAKE for globals (`LAT_MIN`, `RADIUS_KM`)  
- **Docstrings**: Google style; module doc at top  
- **Error handling**: use `retry()` helper for network/API calls  

## 6. Agent Configuration
- **System Prompts** defined in code comments; override via `AGENTS.md` as needed  
- **Model**: default to `gpt-4o-mini` for vision OCR  
- **Retries**: exponential backoff with jitter (3 attempts)  
- **Locale**: Estonian prompts; CLI responses in Estonian/English as coded  

## 7. Web & API Policies
- **Network**: allow outbound to OpenAI and image URLs  
- **Rate limits**: respect OpenAI quotas; exponential backoff on failures  
- **Credentials**: read `OPENAI_API_KEY` from env; never log it  

## 8. Sandbox & Security
- **File I/O**: only read images via paths; no unrestricted writes  
- **Temporary files**: use `tempfile` for any intermediates  
- **Secrets**: no hard-coded tokens; use environment variables  

## 9. Agent Behaviors & Patterns
- **OCR flow**:
  1. Tesseract → parse with `parse_coords`  
  2. Fallback `gpt_vision_ocr` if needed  
- **SALUTE dialog**:
  - Sequential prompts; support “vahele” (skip) and “tagasi” (rewind)  
- **Speak**: all “TTS” via stdout prefix `[SPEAK]`  
- **Exit codes**: use `sys.exit()` on fatal errors  

## 10. Logging & Debugging
- **Stdout**: primary communication channel  
- **Stderr**: warnings and errors prefixed (`⚠️`, `❌`)  
- **Verbose mode**: add `--verbose` flag (future) to show debug logs  

## 11. CI/CD Integration
- **GitHub Actions** (example `.github/workflows/ci.yml`):
  ```yaml
  name: CI
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        - name: Setup Python
          uses: actions/setup-python@v4
          with: python-version: 3.10
        - run: pip install -r requirements.txt
        - run: pytest --maxfail=1 --disable-warnings -q
        - run: flake8 . && black --check .
  ```
- **Docker** (optional): include Dockerfile wrapping above steps

## 12. Examples & Usage
```bash
# OCR image and SALUTE dialogue
./multimodal_cli.py --image drone_view.jpg

# OCR only, skip SALUTE
./multimodal_cli.py --image map.png --no-salute
```
