import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st

# Import centralized paths from config.py
from config import DB_PATH, IMG_FOLDER

# --------------------------------------------------
# Configuration & Setup
# --------------------------------------------------
st.set_page_config(
    page_title="SROIE Receipt Intelligence Dashboard",
    page_icon="🧾",
    layout="wide",
)

# --------------------------------------------------
# Database Connection
# --------------------------------------------------
@st.cache_resource
def get_connection():
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)

if not DB_PATH.exists():
    st.warning(f"No database found at {DB_PATH}. Run `python pipeline.py` first, then refresh this page.")
    st.stop()

conn = get_connection()

# --------------------------------------------------
# Sidebar Navigation & Filters
# --------------------------------------------------
st.sidebar.title("🧾 Receipt Navigator")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Search & Inspect Receipts", "Failed Records Manager"])

# Fetch all data for analytics and tables
try:
    df_all = pd.read_sql("SELECT * FROM parsed_receipts", conn)
except Exception as e:
    st.warning(f"Could not read results ({e}). Run `python pipeline.py` first, then refresh this page.")
    st.stop()

# --------------------------------------------------
# Page 1: Dashboard Overview
# --------------------------------------------------
if page == "Dashboard Overview":
    st.title("📊 SROIE Receipt Intelligence Overview")
    st.markdown("High-level metrics and distribution of your parsed receipt dataset.")

    total_records = len(df_all)
    completed_df = df_all[df_all["status"] == "completed"]
    failed_df = df_all[df_all["status"] == "failed"]
    
    total_spent = completed_df["price"].sum() if "price" in completed_df else 0
    avg_spent = completed_df["price"].mean() if len(completed_df) > 0 and "price" in completed_df else 0
    success_rate = (len(completed_df) / total_records * 100) if total_records > 0 else 0

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Receipts", total_records)
    col2.metric("Total Spending", f"RM {total_spent:,.2f}")
    col3.metric("Average Receipt", f"RM {avg_spent:,.2f}")
    col4.metric("Success Rate", f"{success_rate:.1f}%")

    st.markdown("---")

    # Charts
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Top Vendors by Receipt Count")
        if not completed_df.empty and "company" in completed_df:
            top_vendors = completed_df["company"].value_counts().head(10)
            st.bar_chart(top_vendors)
        else:
            st.info("No completed records found.")

    with col_b:
        st.subheader("Status Breakdown")
        status_counts = df_all["status"].value_counts()
        st.bar_chart(status_counts)

# --------------------------------------------------
# Page 2: Search & Inspect Receipts
# --------------------------------------------------
elif page == "Search & Inspect Receipts":
    st.title("🔍 Search & Inspect Receipts")
    st.markdown("Filter through your parsed receipts, view extracted fields, check OCR text, and inspect images.")

    completed_df = df_all[df_all["status"] == "completed"]
    
    search_query = st.text_input("Search by Company Name or Address", "")
    filtered_df = completed_df
    
    if search_query and not completed_df.empty:
        filtered_df = completed_df[
            completed_df["company"].str.contains(search_query, case=False, na=False) |
            completed_df["address"].str.contains(search_query, case=False, na=False)
        ]

    st.dataframe(
        filtered_df[["filename", "company", "date", "price"]],
        width="stretch",
    )

    st.markdown("---")
    st.subheader("Receipt Detail Inspector")

    filenames = filtered_df["filename"].tolist() if not filtered_df.empty else []
    selected_filename = st.selectbox("Select a receipt file to inspect details:", options=filenames)

    if selected_filename:
        row = filtered_df[filtered_df["filename"] == selected_filename].iloc[0]
        
        col_info, col_img = st.columns(2)

        with col_info:
            st.markdown(f"### 📄 {row['filename']}")
            st.text_input("Company Name", value=str(row["company"]), disabled=True)
            st.text_input("Date", value=str(row["date"]), disabled=True)
            st.text_input("Total Price (RM)", value=str(row["price"]), disabled=True)
            st.text_area("Address", value=str(row["address"]), disabled=True, height=70)
            st.text_area("Raw OCR Text", value=str(row["raw_ocr_text"]), disabled=True, height=150)

        with col_img:
            st.markdown("### 🖼️ Original Receipt Image")
            img_path = IMG_FOLDER / selected_filename
            if img_path.exists():
                st.image(str(img_path), caption=selected_filename, width="stretch")
            else:
                st.warning(f"Image not found at {img_path}")

# --------------------------------------------------
# Page 3: Failed Records Manager
# --------------------------------------------------
elif page == "Failed Records Manager":
    st.title("⚠️ Failed Records Manager")
    st.markdown("Review receipts that encountered errors during processing.")

    failed_df = df_all[df_all["status"] == "failed"]
    st.metric("Total Failed", len(failed_df))

    if not failed_df.empty:
        st.dataframe(
            failed_df[["filename", "error_message"]],
            width="stretch",
        )
    else:
        st.success("🎉 Zero failed records! Your pipeline ran cleanly.")