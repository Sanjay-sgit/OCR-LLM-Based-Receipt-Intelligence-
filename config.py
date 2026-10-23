import os
from pathlib import Path
import dotenv
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
dotenv.load_dotenv(BASE_DIR / ".env")

DB_PATH = BASE_DIR / "extracted_db" / "sroie_results.db"
IMG_FOLDER = BASE_DIR / "dataset_sroie" / "train" / "img"

GEMINI_MODEL = "gemma-4-26b-a4b-it"
API_KEY = os.getenv("API_KEY")


def require_api_key() -> str:
    """Only the pipeline needs the key; the dashboard must open without it."""
    if not API_KEY:
        raise ValueError("No API key found. Add a line API_KEY=your_key to the .env file in the project folder.")
    return API_KEY


class SROIEReceiptSchema(BaseModel):
    company: str | None = Field(default=None, description="Store or company name")
    date: str | None = Field(default=None, description="Date printed on receipt")
    address: str | None = Field(default=None, description="Store address")
    price: str | None = Field(default=None, description="Final total amount paid")