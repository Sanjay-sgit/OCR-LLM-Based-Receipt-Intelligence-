import os
import time
import logging
import argparse
import multiprocessing as mp
import dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pydantic import BaseModel, Field
from google import genai

from config import IMG_FOLDER, GEMINI_MODEL, require_api_key, SROIEReceiptSchema
from database import DatabaseManager
dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

# --------------------------------------------------
# Speed & Rate Settings
# --------------------------------------------------
CPU_COUNT = os.cpu_count() or 4
DEFAULT_WORKERS = max(2, CPU_COUNT // 2)
LLM_BATCH_SIZE = 3                       # 10 receipts per API call
MAX_RPM = 4                            # Strictly capped below free tier limit to prevent 429s

# Global worker OCR instance
_ocr = None

def _init_ocr_worker():
    global _ocr
    from paddleocr import PaddleOCR
    _ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        enable_mkldnn=False
    )

def _ocr_one(img_path: str):
    import cv2
    name = os.path.basename(img_path)
    start = time.perf_counter()
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        texts = []
        for res in _ocr.predict(img):
            if isinstance(res, dict) and "rec_texts" in res:
                texts.extend(res["rec_texts"])
            elif isinstance(res, list):
                for line in res:
                    if len(line) >= 2 and len(line[1]) >= 1:
                        texts.append(str(line[1][0]))
        return name, "\n".join(texts), time.perf_counter() - start, None
    except Exception as e:
        return name, None, time.perf_counter() - start, str(e)


# --------------------------------------------------
# Stage 1: Fast Parallel OCR Runner
# --------------------------------------------------
def run_ocr_stage(images, workers):
    db = DatabaseManager()
    checkpoints = db.get_all_checkpoints()
    
    need_ocr = [p for p in images if not checkpoints.get(p.name) or not checkpoints[p.name].get("raw_ocr_text")]
    
    if not need_ocr:
        logging.info("All images already have cached OCR text. Skipping OCR stage.")
        db.close()
        return

    logging.info(f"--- STAGE 1: Running Parallel OCR for {len(need_ocr)} images using {workers} processes ---")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_ocr_worker) as pool:
        futures = [pool.submit(_ocr_one, str(p)) for p in need_ocr]
        for done, fut in enumerate(as_completed(futures), start=1):
            name, text, secs, err = fut.result()
            if err:
                db.mark_failed(name, err)
            else:
                db.save_ocr_text(name, text)
            if done % 10 == 0 or done == len(need_ocr):
                logging.info(f"OCR Progress: {done}/{len(need_ocr)} completed.")
    db.close()


# --------------------------------------------------
# Stage 2: Controlled Rate-Limited LLM Extractor
# --------------------------------------------------
class BatchReceipt(SROIEReceiptSchema):
    receipt_id: int = Field(description="The receipt number corresponding to the prompt index")

class BatchResponse(BaseModel):
    receipts: list[BatchReceipt]


def run_llm_stage(images):
    db = DatabaseManager()
    try:
        checkpoints = db.get_all_checkpoints()

        pending = [
            (p.name, checkpoints[p.name]["raw_ocr_text"])
            for p in images
            if p.name in checkpoints
            and checkpoints[p.name].get("status") != "completed"
            and checkpoints[p.name].get("raw_ocr_text")
        ]

        if not pending:
            logging.info("No pending LLM extractions found.")
            return

        logging.info(
            f"Retrying {len(pending)} unfinished receipts individually."
        )

        client = genai.Client(api_key=require_api_key())
        interval = 60.0 / MAX_RPM

        for index, (filename, raw_text) in enumerate(pending, start=1):
            # Recheck status to avoid overwriting completed records.
            current = db.get_all_checkpoints().get(filename, {})
            if current.get("status") == "completed":
                continue

            logging.info(
                f"Receipt {index}/{len(pending)}: {filename}"
            )

            prompt = f"""
Extract these fields from this receipt:
- company
- date
- address
- final total price

Return JSON matching SROIEReceiptSchema.
Use null when a field cannot be found.
Do not guess or invent values.

Receipt OCR text:
{raw_text}
"""

            success = False

            for attempt in range(1, 5):
                try:
                    start = time.time()

                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt,
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": SROIEReceiptSchema,
                            "temperature": 0.1,
                        },
                    )

                    if not response.text:
                        raise ValueError("Model returned an empty response")

                    data = SROIEReceiptSchema.model_validate_json(
                        response.text
                    )

                    db.save_record(filename, data, raw_text)
                    logging.info(f"Completed: {filename}")
                    success = True

                    elapsed = time.time() - start
                    if elapsed < interval:
                        time.sleep(interval - elapsed)
                    break

                except Exception as e:
                    message = str(e)
                    logging.warning(
                        f"{filename}: attempt {attempt}/4 failed: "
                        f"{message[:250]}"
                    )

                    if attempt < 4:
                        # Exponential backoff; longer waits for rate limits.
                        lower = message.lower()
                        if "429" in lower or "resource_exhausted" in lower:
                            wait = 60 * attempt
                        else:
                            wait = 15 * (2 ** (attempt - 1))

                        logging.info(f"Waiting {wait}s before retry.")
                        time.sleep(wait)

            if not success:
                logging.error(
                    f"Could not extract {filename}; preserving OCR text."
                )
                db.mark_failed(
                    filename,
                    "LLM extraction failed after 4 attempts; OCR text preserved."
                )

    finally:
        db.close()

    logging.info("LLM extraction stage complete.")


# --------------------------------------------------
# Main Orchestrator
# --------------------------------------------------
def run(
    limit: int | None = None,
    workers: int = DEFAULT_WORKERS,
    skip_ocr: bool = False,
):
    t0 = time.perf_counter()

    images = sorted([
        p for p in IMG_FOLDER.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    if limit is not None:
        images = images[:limit]

    if skip_ocr:
        logging.info("Skipping local OCR; using cached OCR text.")
    else:
        run_ocr_stage(images, workers)

    run_llm_stage(images)

    elapsed = (time.perf_counter() - t0) / 60
    logging.info("Pipeline finished in %.1f minutes!", elapsed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N images")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of OCR worker processes")
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip local OCR and process cached OCR text only",
    )
    args = parser.parse_args()
    run(
        limit=args.limit,
        workers=args.workers,
        skip_ocr=args.skip_ocr,
    )
