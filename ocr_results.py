import csv
import logging
from pathlib import Path

from config import IMG_FOLDER
from database import DatabaseManager

CSV_PATH = Path(__file__).resolve().parent / "ocr_results.csv"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"OCR results not found: {CSV_PATH}")

    if not IMG_FOLDER.exists():
        raise FileNotFoundError(f"Image folder not found: {IMG_FOLDER}")

    valid_names = {
        p.name
        for p in IMG_FOLDER.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    db = DatabaseManager()
    imported = 0
    skipped = 0

    try:
        with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required = {"filename", "raw_ocr_text"}

            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    "CSV must contain filename and raw_ocr_text columns."
                )

            for row in reader:
                filename = (row.get("filename") or "").strip()
                raw_text = row.get("raw_ocr_text") or ""

                if (
                    not filename
                    or filename not in valid_names
                    or not raw_text.strip()
                ):
                    skipped += 1
                    continue

                db.save_ocr_text(filename, raw_text)
                imported += 1

                if imported % 50 == 0:
                    logging.info("Imported %d OCR records", imported)

        logging.info(
            "Import finished: imported=%d, skipped=%d",
            imported,
            skipped,
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
