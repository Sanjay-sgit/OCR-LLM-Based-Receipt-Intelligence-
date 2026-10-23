# OCR Receipt Extraction (SROIE)
Extracts company, date, address and total from scanned receipts using
PaddleOCR + Gemini, stores results in SQLite, and shows them in a Streamlit dashboard.

## Setup
1. Python 3.12: `python -m venv venv` then `venv\Scripts\activate`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your Gemini API key
4. Download the SROIE dataset and place receipt images in `dataset_sroie/train/img/`

## Run
- Dashboard: `streamlit run app.py`
- Full pipeline: `python pipeline.py`