import json
import re
import fitz
from .config import HANDBOOK_PATH, PAGES_JSON, PAGE_IMAGE_DIR

FOOTER = re.compile(r"Approved in the 34th Academic Council Meeting held on 14th August,\s*2024")
PAGENUM = re.compile(r"^\s*\d{1,3}\s*\n?", re.M)
DOTLEADERS = re.compile(r"\.{3,}")  # table-of-contents dot leaders


def extract_handbook():
    PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(HANDBOOK_PATH)
    records = []

    for i, page in enumerate(doc, start=1):
        text = FOOTER.sub("", page.get_text())
        text = PAGENUM.sub("", text, count=1)
        text = DOTLEADERS.sub(" ", text).strip()
        if not text:
            text = f"Page {i}: cover/decorative page of the VU Student Handbook."

        img_path = PAGE_IMAGE_DIR / f"page_{i:02d}.png"
        page.get_pixmap(dpi=200).save(img_path)

        records.append({"page": i, "text": text, "image_path": str(img_path)})

    # the footer we stripped carries the one fact (approval date) that repeats on every page
    records.insert(0, {
        "page": 0,
        "text": "Document: VU Student Handbook, Version 1.0. Approved in the 34th Academic Council Meeting held on 14th August, 2024.",
        "image_path": None,
    })

    PAGES_JSON.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return records


if __name__ == "__main__":
    records = extract_handbook()
    print("Processed", len(records), "pages")
    print("Saved:", PAGES_JSON)
