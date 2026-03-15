import pdfplumber # pyright: ignore[reportMissingImports]
import json
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

COMPANIES = {
    "SHOP_2025": {"company": "Shopify",  "ticker": "SHOP", "year": 2025},
    "RBC_2025":   {"company": "RBC",      "ticker": "RBC",   "year": 2025},
    "TD_2025":   {"company": "TD Bank",  "ticker": "TD",   "year": 2025},
    "SU_2025":   {"company": "Suncor",   "ticker": "SU",   "year": 2025},
    "CNR_2024":  {"company": "CN Rail",  "ticker": "CNR",  "year": 2024},
    "BMO_2025":  {"company": "BMO",      "ticker": "BMO",  "year": 2025},
    "CIBC_2025":   {"company": "CIBC",     "ticker": "CIBC",   "year": 2025},
}

def extract_text_from_pdf(pdf_path: Path) -> list:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"    {total} pages found")
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 100:
                pages.append({
                    "page_number": i + 1,
                    "text": text.strip(),
                })
    return pages


def process_all():
    print("--- ingest.py started ---")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Looking in: {RAW_DATA_DIR.resolve()}")

    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
    print(f"PDFs found: {pdf_files}")

    if not pdf_files:
        print("No PDFs found in data/raw/ — add your PDFs first.")
        return

    for pdf_path in pdf_files:
        stem = pdf_path.stem
        print(f"\nFound file: {stem}")

        if stem not in COMPANIES:
            print(f"  Skipping {pdf_path.name} — not in COMPANIES dict")
            continue

        meta = COMPANIES[stem]
        print(f"Processing {meta['company']} ({stem}.pdf)...")

        pages = extract_text_from_pdf(pdf_path)
        print(f"    {len(pages)} usable pages extracted")

        output = {
            "company": meta["company"],
            "ticker":  meta["ticker"],
            "year":    meta["year"],
            "source":  pdf_path.name,
            "pages":   pages,
            "total_pages_extracted": len(pages),
        }

        out_path = PROCESSED_DATA_DIR / f"{stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        total_chars = sum(len(p["text"]) for p in pages)
        print(f"    Saved to {out_path} ({total_chars:,} total characters)")

    print("\n--- ingest.py finished ---")


if __name__ == "__main__":
    process_all()