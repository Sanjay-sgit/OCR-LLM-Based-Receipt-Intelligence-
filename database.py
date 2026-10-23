import re
import sqlite3
from pathlib import Path
from config import DB_PATH, SROIEReceiptSchema

class DatabaseManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Using check_same_thread=False allows safe thread utilization with local connections
        self.conn = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS parsed_receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                company TEXT,
                date TEXT,
                address TEXT,
                price REAL,
                raw_ocr_text TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                error_message TEXT
            )
        """)
        self.conn.commit()

    def get_checkpoint(self, filename: str) -> dict | None:
        row = self.conn.execute(
            "SELECT filename, raw_ocr_text, status FROM parsed_receipts WHERE filename = ?",
            (filename,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_checkpoints(self) -> dict[str, dict]:
        """One query for every file, so a re-run knows what to skip."""
        rows = self.conn.execute("SELECT filename, raw_ocr_text, status FROM parsed_receipts").fetchall()
        return {r["filename"]: dict(r) for r in rows}

    def save_ocr_text(self, filename: str, raw_text: str):
        """Cache OCR output without resetting completed records."""
        self.conn.execute("""
            INSERT INTO parsed_receipts
                (filename, raw_ocr_text, status, error_message)
            VALUES (?, ?, 'ocr_done', NULL)
            ON CONFLICT(filename) DO UPDATE SET
                raw_ocr_text = excluded.raw_ocr_text,
                status = CASE
                    WHEN parsed_receipts.status = 'completed'
                    THEN 'completed'
                    ELSE 'ocr_done'
                END,
                error_message = CASE
                    WHEN parsed_receipts.status = 'completed'
                    THEN parsed_receipts.error_message
                    ELSE NULL
                END
        """, (filename, raw_text))
        self.conn.commit()

    def save_record(self, filename: str, data: SROIEReceiptSchema, raw_text: str):
        price_val = None
        if data.price:
            matches = re.findall(r"\d+(?:,\d{3})*(?:\.\d{1,2})?", data.price)
            if matches:
                try:
                    price_val = float(matches[-1].replace(",", ""))
                except ValueError:
                    pass

        self.conn.execute("""
            INSERT INTO parsed_receipts (filename, company, date, address, price, raw_ocr_text, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, 'completed', NULL)
            ON CONFLICT(filename) DO UPDATE SET
                company = excluded.company,
                date = excluded.date,
                address = excluded.address,
                price = excluded.price,
                raw_ocr_text = excluded.raw_ocr_text,
                status = 'completed',
                error_message = NULL
        """, (filename, data.company, data.date, data.address, price_val, raw_text))
        self.conn.commit()

    def mark_failed(self, filename: str, error: str):
        self.conn.execute("""
            INSERT INTO parsed_receipts (filename, status, error_message)
            VALUES (?, 'failed', ?)
            ON CONFLICT(filename) DO UPDATE SET status = 'failed', error_message = ?
        """, (filename, error[:500], error[:500]))
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()