#!/usr/bin/env python3
"""
multimodal_cli.py  –  May-2025 (Estonian voice edition)

1. Image → GPS (Tesseract first, GPT-4o-mini fallback)
2. Estonia + 500 km sanity, MGRS conversion
3. Mic → Whisper (language='et')   → SALUTE prompts
4. Speech:
      • printed messages only (no external TTS engines)

Modified for testing without audio: user responds via stdin.
"""

import argparse
import base64
import os
import random
import re
import sys
import tempfile
import time
import math
import shutil
from mimetypes import guess_type
from io import BytesIO
# import wave

# ───────── third-party ─────────
from openai import OpenAI
import requests
import pytesseract
from PIL import Image, ImageEnhance
# import sounddevice as sd
# import numpy as np
from mgrs import MGRS

# ─────────── constants ───────────
LAT_MIN, LAT_MAX = 55.0, 60.0
LON_MIN, LON_MAX = 21.0, 28.0
REF_LAT, REF_LON = 58.0, 25.0
RADIUS_KM = 500.0

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    sys.exit("❌  Set OPENAI_API_KEY in your environment.")
client = OpenAI(api_key=OPENAI_API_KEY)

if shutil.which("tesseract") is None:
    print("⚠️  Tesseract not in PATH – OCR fallback will fail", file=sys.stderr)

# ───────── retry utility ─────────
def retry(fn, *args, retries=3, **kwargs):
    for i in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if i == retries:
                raise
            back = 2 ** i + random.random()
            print(f"[Retry {i}/{retries}] {fn.__name__}: {e} → {back:.1f}s")
            time.sleep(back)

# ───────── helpers ─────────
def to_data_url(path: str) -> str:
    mime, _ = guess_type(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime or 'application/octet-stream'};base64,{b64}"

def _haversine(a1,a2,b1,b2):
    R=6371
    dlat=math.radians(b1-a1)
    dlon=math.radians(b2-a2)
    h=(math.sin(dlat/2)**2 +
       math.cos(math.radians(a1))*
       math.cos(math.radians(b1))*
       math.sin(dlon/2)**2)
    return R*2*math.asin(math.sqrt(h))

# ───────── OCR pipeline ─────────
def preprocess(img):
    return ImageEnhance.Contrast(img.convert("L")).enhance(2.0)

def tesseract_ocr(path):
    try:
        base = Image.open(path)
    except Exception as e:
        print("❌  Cannot open image:", e, file=sys.stderr)
        return ""
    out = ""
    for angle in (0,90,180,270):
        txt = pytesseract.image_to_string(
            preprocess(base.rotate(angle, expand=True)),
            config="--psm 6 -c tessedit_char_whitelist=0123456789.,NSEW "
        ).strip()
        print(f"\n[Tesseract {angle}°]\n{txt or '[no text]'}")
        out = txt or out
        if ("N" in txt or "S" in txt) and ("E" in txt or "W" in txt):
            if parse_coords(txt):
                return txt
    return out

def gpt_vision_ocr(path):
    resp = retry(
        client.responses.create,
        model="gpt-4o-mini",
        input=[{
            "role": "user",
            "content": [
                {"type":"input_text", "text":
                 "Leia drooni ekraanil kuvatud GPS-koordinaadid ja vasta ainult kümnend‑kraadides kujul, "
                 "näiteks: 57.927336° N, 26.747699° E"},
                {"type":"input_image","image_url":to_data_url(path)}
            ]
        }],
        timeout=60
    )
    return resp.output_text.strip()

# ───────── coordinate utils ─────────
def parse_coords(txt):
    if not txt:
        return None
    t = txt.upper().replace("°","")
    lat = lon = None
    m1 = re.search(r"(-?\d+\.\d+)\s*([NS])", t)
    m2 = re.search(r"(-?\d+\.\d+)\s*([EW])", t)
    if m1 and m2:
        lat = float(m1.group(1)) * (1 if m1.group(2)=="N" else -1)
        lon = float(m2.group(1)) * (1 if m2.group(2)=="E" else -1)
    else:
        parts = re.split(r"[,\s]+", re.sub(r"[NSEW]", "", t).strip())
        if len(parts) >= 2:
            try:
                lat, lon = float(parts[0]), float(parts[1])
            except:
                return None
    if lat is None or lon is None:
        return None
    # swap if out of bounds
    if not (LAT_MIN<=lat<=LAT_MAX) and (LAT_MIN<=lon<=LAT_MAX):
        lat,lon = lon,lat
    if not (LAT_MIN<=lat<=LAT_MAX and LON_MIN<=lon<=LON_MAX):
        return None
    if _haversine(lat,lon,REF_LAT,REF_LON) > RADIUS_KM:
        print("⚠️  >500 km kaugusel kontrollpunktist.")
    return lat, lon

# ───────── "TTS" as print ─────────
def speak(text):
    print(f"[SPEAK] {text}")

# ───────── Whisper STT (stdin) ─────────
# Original audio recording disabled. Instead, prompt user for text.
def record_and_transcribe(sec=5):
    return input("[STT] (type response) ")

# ───────── MGRS helpers ─────────
def to_mgrs(lat,lon): return MGRS().toMGRS(lat,lon)
def format_mgrs(m):
    z = m[:5]; r = m[5:]; h = len(r)//2
    return f"{z} {r[:h]} {r[h:]}"

# ───────── SALUTE (Estonian prompts) ─────────
def salute_dialogue(loc_str):
    fields = [
        ("size","Suurus"),
        ("activity","Tegevus"),
        ("location","Asukoht"),
        ("unit","Üksus"),
        ("time","Aeg"),
        ("equipment","Varustus")
    ]
    ans = {k:"" for k,_ in fields}
    ans["location"] = loc_str
    idx = 0
    while idx < len(fields):
        key, et = fields[idx]
        print(f"[SALUTE_PROMPT] Palun ütle {et}. (või 'vahele'/'tagasi')")
        txt = record_and_transcribe()
        low = txt.lower()
        if low.startswith("tagasi"):
            parts = low.split()
            if len(parts) >= 2:
                tgt = parts[1]
                for j,(k,_) in enumerate(fields):
                    if tgt == k or tgt == _.lower():
                        idx = j
                        break
            continue
        if low in ("vahele","skip","järgmine"):
            ans[key] = ""
        else:
            ans[key] = txt
        idx += 1

    print("[SALUTE_REPORT]")
    for k,et in fields:
        print(f"{et}: {ans[k] or '–'}")

# ───────── main ─────────
def main():
    ap = argparse.ArgumentParser(description="Estonian multimodal CLI")
    ap.add_argument("--image", required=True, help="Pilt GPS-koordinaatidega")
    ap.add_argument("--no-salute", action="store_true",
                    help="Ära käivita SALUTE dialoogi")
    args = ap.parse_args()

    # 1) Tesseract OCR
    raw_ocr = tesseract_ocr(args.image)
    print("\n=== Tesseract OCR ===\n", raw_ocr or "[tühi]")

    # 2) parse
    coords = parse_coords(raw_ocr)

    if coords:
        print("[Info] OCR found valid coordinates; skipping LLM OCR.")
    else:
        # 3) fallback to GPT vision OCR
        try:
            raw_llm = gpt_vision_ocr(args.image)
            print("\n=== GPT-4o-mini ===\n", raw_llm or "[tühi]")
            coords = parse_coords(raw_llm)
        except Exception as e:
            print(f"[LLM error] {e}", file=sys.stderr)

    # 4) bail if still none
    if not coords:
        sys.exit("❌  Koordinaate ei saanud kätte.")

    # 5) output
    lat, lon = coords
    dec = f"{abs(lat):.6f}° {'N' if lat>=0 else 'S'}, {abs(lon):.6f}° {'E' if lon>=0 else 'W'}"
    mgrs = format_mgrs(to_mgrs(lat, lon))
    print(f"\n[RESULT] Koordinaadid leitud. {dec}")
    print(f"[MGRS] {mgrs}")

    speak(f"Koordinaadid leitud. {dec}")

    if not args.no_salute:
        salute_dialogue(mgrs)

if __name__ == "__main__":
    main()
