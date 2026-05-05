import fitz
import torch
import json
import re
import os
import pandas as pd
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# ==============================
# SETUP
# ==============================
OUTPUT_FOLDER = "output_tables"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model_name = "openbmb/MiniCPM-V-2_6"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True
)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# Auto device selection
if torch.cuda.is_available():
    model = model.cuda()
elif torch.backends.mps.is_available():
    model = model.to("mps")
else:
    model = model.float()   # CPU fallback
model.eval()

# ==============================
# PROMPT (replaces Steps 2-8)
# ==============================
PROMPT = """You are a precise table extractor.
Extract ALL tables from this document image.

Return ONLY this JSON format:
{
  "tables": [
    {
      "title": "table title",
      "headers": ["col1", "col2", ...],
      "rows": [
        ["val1", "val2", ...],
        ["val1", "val2", ...]
      ]
    }
  ]
}

STRICT RULES:
- Preserve numbers exactly: decimals, negatives (-1.77), zeros (0.00)
- Keep row hierarchy: main rows AND sub-rows (a, b, c...)
- Include ALL rows — do not skip any
- Multi-level headers: join with ' > ' e.g. 'Area Sown > 2024'
- Footnotes marked * or **: include as last row
- Return ONLY valid JSON
"""

# ==============================
# MAIN PIPELINE
# (replaces your entire pipeline)
# ==============================
def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    print(f"Total pages: {len(doc)}")

    for page_num in range(len(doc)):
        print(f"\nProcessing page {page_num + 1}...")

        # Render page
        page = doc[page_num]
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes(
            "RGB", [pix.width, pix.height], pix.samples
        )

        # MiniCPM-V replaces ALL OpenCV + OCR + DBSCAN steps
        msgs = [{"role": "user", "content": [img, PROMPT]}]

        response = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.1,
            max_new_tokens=4096
        )

        # Parse response
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found")

            data = json.loads(json_match.group())
            tables = data.get("tables", [])

            for j, table in enumerate(tables):
                headers = table.get("headers", [])
                rows    = table.get("rows", [])

                if not rows:
                    print(f"  Table {j+1}: empty — skipping")
                    continue

                df = pd.DataFrame(rows, columns=headers if headers else None)

                out_path = os.path.join(
                    OUTPUT_FOLDER,
                    f"page_{page_num+1}_table_{j+1}.csv"
                )
                df.to_csv(out_path, index=False)
                print(f"  ✅ Table {j+1} saved: "
                      f"{df.shape[0]} rows x {df.shape[1]} cols → {out_path}")

        except Exception as e:
            print(f"  ⚠️ Page {page_num+1} parse error: {e}")
            # Save raw response for debugging
            raw_path = os.path.join(
                OUTPUT_FOLDER,
                f"page_{page_num+1}_raw.txt"
            )
            with open(raw_path, "w") as f:
                f.write(response)

    print("\n✅ All pages processed!")

# ==============================
# RUN
# ==============================
process_pdf("IMAGE_PDF.pdf")
