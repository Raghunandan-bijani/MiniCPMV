#!/usr/bin/env python3
"""
End-to-end PDF Table Extractor using MiniCPM-V via Ollama
- Converts PDF pages to images
- Sends each page to MiniCPM-V with a robust prompt
- Parses JSON table responses
- Saves all extracted tables to an Excel file
"""

import base64
import json
import re
import io
import requests
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
import os
import time
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION — edit these as needed
# ─────────────────────────────────────────────
PDF_PATH       = "IMAGE_PDF.pdf"          # Path to your input PDF
OUTPUT_EXCEL   = "extracted_tables.xlsx"  # Output Excel file
OLLAMA_URL     = "http://localhost:11434/api/generate"
MODEL_NAME     = "minicpm-v"              # Ollama model name
DPI            = 200                      # Resolution for PDF-to-image conversion
# ─────────────────────────────────────────────


SYSTEM_PROMPT = """You are an expert document analyst and OCR engine.
Your job is to carefully examine document images — including scanned,
rotated, or low-quality pages — and extract all tabular data accurately.

Rules you must follow:
- Auto-correct any rotation in the image before reading.
- Fully extract ALL tables present on the page.
- Preserve multi-line column headers as a single joined string (e.g. "Crop / Season").
- Keep numeric values exactly as they appear (do not round or alter).
- Return ONLY valid JSON — no explanations, no markdown, no extra text.
"""

USER_PROMPT = """Look at this document page carefully.

1. Detect and mentally correct any page rotation first.
2. Identify ALL tables on the page.
3. For each table, extract:
   - "table_title": a short descriptive title (use "" if none visible)
   - "headers": list of column header strings (join multi-line headers with " / ")
   - "rows": list of rows, where each row is a list of cell values (as strings)

Return ONLY a JSON object in this exact format:
{
  "tables": [
    {
      "table_title": "...",
      "headers": ["col1", "col2", ...],
      "rows": [
        ["val1", "val2", ...],
        ...
      ]
    }
  ]
}

If no table is found on the page, return: {"tables": []}
"""


def image_to_base64(pil_image: Image.Image) -> str:
    """Convert a PIL image to a base64-encoded JPEG string."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_json_from_response(text: str) -> dict:
    """Robustly parse JSON from model response, even if wrapped in markdown."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.rstrip("`").strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the response
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Return empty result if parsing fails
    print("  [WARNING] Could not parse JSON from model response.")
    print(f"  Raw response snippet: {text[:300]}")
    return {"tables": []}


def query_minicpm(image_b64: str, page_num: int) -> dict:
    """Send a page image to MiniCPM-V via Ollama and return parsed JSON."""
    payload = {
        "model": MODEL_NAME,
        "system": SYSTEM_PROMPT,
        "prompt": USER_PROMPT,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,    # Low temperature for factual extraction
            "num_predict": 4096    # Allow long responses for large tables
        }
    }

    try:
        print(f"    Sending to Ollama...")
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        raw_text = result.get("response", "")
        return extract_json_from_response(raw_text)

    except requests.exceptions.ConnectionError:
        print("  [ERROR] Cannot connect to Ollama. Is it running? (ollama serve)")
        return {"tables": []}
    except requests.exceptions.Timeout:
        print("  [ERROR] Request timed out. Try increasing timeout or reducing DPI.")
        return {"tables": []}
    except Exception as e:
        print(f"  [ERROR] Unexpected error on page {page_num}: {e}")
        return {"tables": []}


def validate_table(table: dict) -> bool:
    """Basic validation — skip malformed or empty tables."""
    headers = table.get("headers", [])
    rows    = table.get("rows", [])
    if not headers and not rows:
        return False
    if len(headers) == 0:
        return False
    return True


def extract_tables_from_pdf(pdf_path: str) -> list:
    """
    Main extraction pipeline:
    1. Convert each PDF page to an image
    2. Send to MiniCPM-V
    3. Collect all tables with page metadata
    """
    print(f"\n[1/3] Converting PDF to images at {DPI} DPI...")
    try:
        pages = convert_from_path(pdf_path, dpi=DPI)
    except Exception as e:
        print(f"  [ERROR] Failed to convert PDF: {e}")
        return []
    
    print(f"      Found {len(pages)} page(s).")

    all_tables = []
    start_time = time.time()

    print(f"\n[2/3] Extracting tables via MiniCPM-V ({MODEL_NAME})...")
    for page_num, page_img in enumerate(pages, start=1):
        page_start = time.time()
        print(f"\n  Page {page_num}/{len(pages)}...")
        
        image_b64 = image_to_base64(page_img)
        result    = query_minicpm(image_b64, page_num)
        tables    = result.get("tables", [])

        valid_tables = [t for t in tables if validate_table(t)]
        elapsed = time.time() - page_start
        print(f"    -> Found {len(valid_tables)} valid table(s). Time: {elapsed:.1f}s")

        for idx, table in enumerate(valid_tables, start=1):
            all_tables.append({
                "page":        page_num,
                "table_index": idx,
                "table_title": table.get("table_title", ""),
                "headers":     table.get("headers", []),
                "rows":        table.get("rows", [])
            })

    total_elapsed = time.time() - start_time
    print(f"\n  Total extraction time: {total_elapsed:.1f}s")
    return all_tables


def save_to_excel(all_tables: list, output_path: str):
    """Save all extracted tables to separate sheets in an Excel file."""
    print(f"\n[3/3] Saving {len(all_tables)} table(s) to '{output_path}'...")

    if not all_tables:
        print("  [WARNING] No tables found. Excel file will not be created.")
        return

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_rows = []

        for entry in all_tables:
            page        = entry["page"]
            table_index = entry["table_index"]
            title       = entry["table_title"]
            headers     = entry["headers"]
            rows        = entry["rows"]

            # Pad or trim rows to match header length
            n_cols = len(headers) if headers else (len(rows[0]) if rows else 1)
            padded_rows = []
            
            for row in rows:
                if len(row) < n_cols:
                    row = list(row) + [""] * (n_cols - len(row))
                elif len(row) > n_cols:
                    row = row[:n_cols]
                padded_rows.append(row)

            if headers:
                df = pd.DataFrame(padded_rows, columns=headers)
            else:
                df = pd.DataFrame(padded_rows)

            # Sheet name: max 31 chars (Excel limit)
            sheet_name = f"P{page}_T{table_index}"
            if title:
                clean_title = re.sub(r'[\\/*?:\[\]]', '', title)[:20]
                sheet_name  = f"P{page}_{clean_title}"[:31]

            try:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"    -> Sheet '{sheet_name}': {len(df)} row(s), {len(df.columns)} col(s)")

                summary_rows.append({
                    "Page":        page,
                    "Table Index": table_index,
                    "Table Title": title if title else "(No title)",
                    "Sheet Name":  sheet_name,
                    "Rows":        len(df),
                    "Columns":     len(df.columns)
                })
            except Exception as e:
                print(f"    [!] Failed to write sheet '{sheet_name}': {e}")

        # Write summary sheet at the beginning
        try:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            print(f"    -> Sheet 'Summary': index of all extracted tables.")
        except Exception as e:
            print(f"    [!] Failed to write summary sheet: {e}")

    print(f"\n  ✓ Done! Saved to: {output_path}")


def main():
    print("=" * 70)
    print("  PDF TABLE EXTRACTOR — MiniCPM-V via Ollama")
    print("=" * 70)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PDF file:  {PDF_PATH}")
    print(f"  Output:    {OUTPUT_EXCEL}")
    print(f"  Model:     {MODEL_NAME}")
    print(f"  Ollama:    {OLLAMA_URL}")
    print("=" * 70)

    # Validate PDF path
    if not os.path.exists(PDF_PATH):
        print(f"\n  [ERROR] PDF not found: '{PDF_PATH}'")
        print("  Please update the PDF_PATH variable at the top of the script.")
        return

    start = time.time()
    all_tables = extract_tables_from_pdf(PDF_PATH)
    
    if all_tables:
        save_to_excel(all_tables, OUTPUT_EXCEL)
        print(f"\n  ✓ SUCCESS: {len(all_tables)} table(s) extracted and saved.")
    else:
        print("\n  ⚠ No tables were found in the document.")

    elapsed = time.time() - start
    print(f"\n  Total time: {int(elapsed // 60)}m {int(elapsed % 60)}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
