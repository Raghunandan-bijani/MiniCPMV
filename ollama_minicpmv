import ollama
import base64
import json
import re
import os
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import io
import time
from datetime import datetime

# ══════════════════════════════════════════════════════════════════
#  OLLAMA + MiniCPM-V  —  PDF Table Extractor
#  Works on : Intel Core Ultra 5  |  16 GB RAM  |  No discrete GPU
#  Requirement : ollama running locally (ollama serve)
# ══════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION  (edit these as needed)
# ──────────────────────────────────────────────────────────────────
PDF_PATH      = "IMAGE_PDF.pdf"          # ← your PDF file path
OUTPUT_FOLDER = "output_tables"          # ← folder for CSVs + Excel
EXCEL_NAME    = "Kharif_Extracted.xlsx"  # ← final Excel file name
OLLAMA_MODEL  = "minicpm-v"              # ← model name in Ollama
DPI           = 200                      # ← page render resolution
MAX_TOKENS    = 4096                     # ← max tokens per response
TEMPERATURE   = 0.1                      # ← low = deterministic output

# ──────────────────────────────────────────────────────────────────
# 1.  PROMPT  (instructs MiniCPM-V exactly what to extract)
# ──────────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """You are a precise agricultural data table extractor.
Carefully look at this document image and extract ALL tables visible.

Return ONLY the following JSON format — no extra text, no markdown:
{
  "tables": [
    {
      "title": "exact table title from the document",
      "headers": ["Column 1", "Column 2", "Column 3"],
      "rows": [
        ["row1_val1", "row1_val2", "row1_val3"],
        ["row2_val1", "row2_val2", "row2_val3"]
      ]
    }
  ]
}

STRICT EXTRACTION RULES:
1.  Preserve ALL numbers exactly as shown
      - Decimals  : 394.28  not  394
      - Negatives : -1.77   not  1.77
      - Zeros     : 0.00    not  blank
2.  Keep row hierarchy intact
      - Main rows  : "1. Total Cereals"
      - Sub rows   : "a. Rice", "b. Wheat", "c. Maize"
3.  Multi-level column headers → join with " > "
      - Example   : "Area Sown > Current Year > Kharif"
4.  If a cell spans multiple columns repeat the value
5.  Include footnotes (lines starting with * or **) as last rows
6.  Do NOT skip any row — extract every single row
7.  Do NOT add any text outside the JSON block
8.  If no table found on page return:  {"tables": []}
"""

# ──────────────────────────────────────────────────────────────────
# 2.  HELPER UTILITIES
# ──────────────────────────────────────────────────────────────────
def banner(msg: str) -> None:
    """Print a clearly visible section banner."""
    width = 64
    print("\n" + "═" * width)
    print(f"  {msg}")
    print("═" * width)


def pil_to_base64(pil_img: Image.Image) -> str:
    """Convert a PIL image to a base-64 PNG string for Ollama."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_json(raw_text: str) -> dict | None:
    """
    Robustly pull the first {...} JSON block out of a raw LLM response.
    Tries strict parse first, then falls back to regex extraction.
    """
    # Attempt 1 — direct parse (model returned clean JSON)
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    # Attempt 2 — find outermost { ... } block
    try:
        start = raw_text.index("{")
        end   = raw_text.rindex("}") + 1
        return json.loads(raw_text[start:end])
    except (ValueError, json.JSONDecodeError):
        pass

    # Attempt 3 — regex for json code block (```json ... ```)
    match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None   # all attempts failed


def safe_sheet_name(raw: str, index: int) -> str:
    """
    Build a valid Excel sheet name (max 31 chars, no special chars).
    """
    name = re.sub(r"[\\/:*?\[\]]", "", str(raw))
    name = name.strip()[:25] or f"Table_{index}"
    return f"{index}_{name}"


def check_ollama_running() -> bool:
    """
    Ping Ollama to make sure the server is up before we start.
    Returns True if reachable, False otherwise.
    """
    try:
        ollama.list()
        return True
    except Exception:
        return False


def check_model_available(model_name: str) -> bool:
    """
    Check whether the requested model is already pulled in Ollama.
    """
    try:
        models = ollama.list()
        # ollama.list() returns a dict with key 'models'
        available = [m["name"] for m in models.get("models", [])]
        return any(model_name in m for m in available)
    except Exception:
        return False

# ──────────────────────────────────────────────────────────────────
# 3.  PDF → PAGE IMAGES
# ──────────────────────────────────────────────────────────────────
def pdf_to_page_images(pdf_path: str, dpi: int = 200) -> list[Image.Image]:
    """
    Render every page of a PDF to a PIL Image at the given DPI.
    Higher DPI = better OCR accuracy but slower processing.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc    = fitz.open(pdf_path)
    images = []

    print(f"  PDF loaded  : {pdf_path}")
    print(f"  Total pages : {len(doc)}")
    print(f"  Render DPI  : {dpi}")

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix  = page.get_pixmap(dpi=dpi)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images

# ──────────────────────────────────────────────────────────────────
# 4.  SINGLE PAGE  →  OLLAMA  →  PARSED TABLES
# ──────────────────────────────────────────────────────────────────
