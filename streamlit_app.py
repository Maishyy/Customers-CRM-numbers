import re
import pandas as pd
import streamlit as st
import pdfplumber
from io import BytesIO
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json
import firebase_admin
from firebase_admin import credentials, firestore
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')]
)
logger = logging.getLogger(__name__)

# --- Firebase Setup ---
try:
    firebase_json = json.loads(st.secrets["firebase_creds"])
    cred = credentials.Certificate(firebase_json)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    logger.error(f"Firebase initialization failed: {str(e)}")
    st.error("Failed to initialize Firebase. Check logs for details.")
    st.stop()

# --- Constants ---
PHONE_REGEX = r"(?:(?:\+254|254|0)?(7\d{8}|11\d{7}))"
NAME_REGEX = r"([A-Z][A-Z]+\s+){1,}[A-Z][A-Z]+"
EXCLUDE_KEYWORDS = ["CDM", "BANK", "TRANSFER", "CHEQUE"]
BLACKLISTED_NUMBERS = ["+254722000000", "+254000000000"]
MAX_FILE_SIZE_MB = 10  # Maximum file size allowed (10MB)

# --- Optimized Helpers ---
def normalize_number(raw):
    """Normalize phone numbers to +254 format with better validation"""
    if not raw or not isinstance(raw, str):
        return None
    
    try:
        cleaned = re.sub(r'[^\d]', '', raw)
        if len(cleaned) == 9 and cleaned.startswith('7'):
            normalized = f"+254{cleaned}"
        elif len(cleaned) == 12 and cleaned.startswith('254'):
            normalized = f"+{cleaned}"
        elif len(cleaned) == 10 and cleaned.startswith('0'):
            normalized = f"+254{cleaned[1:]}"
        else:
            return None
        
        if normalized in BLACKLISTED_NUMBERS or not normalized.startswith("+2547"):
            return None
        return normalized
    except Exception:
        return None

def should_exclude_line(line):
    """Check if line contains excluded keywords"""
    if not isinstance(line, str):
        return True
    return any(keyword in line.upper() for keyword in EXCLUDE_KEYWORDS)

# --- Parallel Processing Functions ---
def process_file(file):
    """Process a single file with appropriate handler"""
    try:
        file_ext = file.name.lower().split('.')[-1]
        
        if file_ext == "pdf":
            return process_pdf(file)
        elif file_ext == "csv":
            return process_csv(file)
        elif file_ext in ("xls", "xlsx"):
            return process_excel(file)
        else:
            st.warning(f"Unsupported file format: {file.name}")
            return []
    except Exception as e:
        logger.error(f"Error processing {file.name}: {str(e)}")
        st.error(f"Error processing {file.name}: {str(e)}")
        return []

def process_files_parallel(files):
    """Process multiple files in parallel"""
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, files))
    return [contact for sublist in results for contact in sublist]

# --- Optimized Extractors ---
@lru_cache(maxsize=32)
def extract_contacts(text):
    """Extract contacts from text with caching"""
    records = []
    for match in re.finditer(PHONE_REGEX, text):
        phone = normalize_number(match.group())
        if not phone:
            continue
            
        after_number = text[match.end():].strip()
        name_match = re.search(NAME_REGEX, after_number)
        name = name_match.group(0).strip() if name_match else ""
        
        records.append((phone, name))
    return records

def process_pdf(file):
    """Optimized PDF processing with page sampling for large files"""
    try:
        with pdfplumber.open(file) as pdf:
            # Sample pages if PDF is large (>20 pages)
            pages = pdf.pages[::2] if len(pdf.pages) > 20 else pdf.pages
            return [contact for page in pages 
                    if (text := page.extract_text())
                    for contact in extract_contacts(text)]
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        return []

def process_csv(file):
    """Optimized CSV processing with chunking"""
    try:
        # Use chunking for large files (>5MB)
        if file.size > 5 * 1024 * 1024:
            chunks = pd.read_csv(file, chunksize=10000)
            return [contact for chunk in chunks 
                    for contact in extract_from_dataframe(chunk)]
        return extract_from_dataframe(pd.read_csv(file))
    except Exception as e:
        logger.error(f"CSV processing error: {str(e)}")
        return []

def process_excel(file):
    """Optimized Excel processing with engine fallback"""
    try:
        try:
            dfs = pd.read_excel(file, sheet_name=None, engine='openpyxl')
        except:
            dfs = pd.read_excel(file, sheet_name=None, engine='xlrd')
        
        # Process only first 3 sheets if many sheets exist
        sheets = list(dfs.items())[:3] if len(dfs) > 3 else dfs.items()
        return [contact for _, df in sheets
                for contact in extract_from_dataframe(df)]
    except Exception as e:
        logger.error(f"Excel processing error: {str(e)}")
        return []

def extract_from_dataframe(df):
    """Extract contacts from DataFrame"""
    records = []
    for _, row in df.iterrows():
        row_text = " ".join(str(x) for x in row if pd.notna(x))
        if not should_exclude_line(row_text):
            records.extend(extract_contacts(row_text))
    return records

# --- Optimized Firebase Operations ---
def save_to_firestore(data):
    """Batch save contacts to Firestore"""
    if not db or not data:
        return 0
    
    collection = db.collection("contacts")
    batch = db.batch()
    new_count = 0
    
    # Deduplicate before processing
    unique_contacts = {(p, n) for p, n in data if p}
    
    for phone, name in unique_contacts:
        doc_ref = collection.document(phone.replace("+", ""))
        
        # Check existence first
        if not doc_ref.get().exists:
            first, last = ("", "")
            if name:
                parts = name.strip().split(" ", 1)
                first = parts[0]
                last = parts[1] if len(parts) > 1 else ""

            batch.set(doc_ref, {
                "phone_number": phone,
                "client_name": name,
                "first_name": first,
                "last_name": last,
                "source": "upload",
                "timestamp": firestore.SERVER_TIMESTAMP
            })
            new_count += 1
            
            # Commit every 500 operations to avoid batch limits
            if new_count % 500 == 0:
                batch.commit()
                batch = db.batch()
    
    if new_count % 500 != 0:
        batch.commit()
    
    return new_count

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Sequid Hardware Contact System")

tabs = st.tabs(["Upload & Sync", "Daily SMS List", "Dashboard"])

# --- Tab 1: Upload ---
with tabs[0]:
    st.subheader("Upload MPESA or Bank Statements")
    uploaded_files = st.file_uploader(
        "Upload files", 
        type=["pdf", "csv", "xls", "xlsx"], 
        accept_multiple_files=True,
        help=f"Max file size: {MAX_FILE_SIZE_MB}MB each"
    )

    if uploaded_files:
        # Check for oversized files
        oversized = [f.name for f in uploaded_files if f.size > MAX_FILE_SIZE_MB * 1024 * 1024]
        if oversized:
            st.error(f"These files exceed size limit: {', '.join(oversized)}")
        else:
            with st.spinner("Processing files..."):
                if len(uploaded_files) > 1:
                    all_data = process_files_parallel(uploaded_files)
                else:
                    all_data = process_file(uploaded_files[0])

            if all_data:
                unique_count = len({p for p, _ in all_data if p})
                st.success(f"Found {unique_count} unique contacts")
                
                new_count = save_to_firestore(all_data)
                st.success(f"Added {new_count} new contacts to Firestore")
                
                excel_file, df = generate_excel(all_data)
                st.download_button(
                    "Download Excel", 
                    data=excel_file, 
                    file_name="contacts.xlsx",
                    help="Download all processed contacts"
                )
                st.dataframe(df.head())
            else:
                st.warning("No valid phone numbers found.")

# [Rest of your tabs remain the same...]

# --- Tab 2: Daily SMS List ---
with tabs[1]:
    st.subheader("Generate Daily SMS List (only message once per 7 days)")
    if st.button("Generate Today's List"):
        sms_data = get_eligible_sms_contacts()
        if sms_data:
            sms_excel, sms_df = generate_excel(sms_data)
            st.success(f"Prepared {len(sms_data)} new contacts for messaging.")
            st.download_button("Download SMS List", data=sms_excel, file_name="todays_sms_list.xlsx")
            st.dataframe(sms_df)
        else:
            st.info("No new contacts to message today.")
            @st.cache_data
def load_message_logs():
    try:
        collection_ref = db.collection("message_logs")
        docs = collection_ref.stream()

        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            phone = doc_data.get("phone_number", "")
            date = doc_data.get("timestamp")

            if isinstance(date, datetime):
                date_formatted = date.strftime("%Y-%m-%d")
            else:
                date_formatted = ""

            data.append({
                "Phone": phone,
                "Date Messaged": date_formatted
            })

        return pd.DataFrame(data)

    except Exception as e:
        logger.error(f"Error loading message logs: {str(e)}")
        st.error("Failed to load message logs.")
        return pd.DataFrame(columns=["Phone", "Date Messaged"])


# --- Tab 3: Dashboard ---
with tabs[2]:
    st.subheader("Message History Dashboard")
    df_logs = load_message_logs()

    if df_logs.empty:
        st.info("No message logs found.")
    else:
        # Metrics
        st.metric("Total Messages Sent", len(df_logs))
        recent = df_logs["Date Messaged"].value_counts().sort_index()
        st.bar_chart(recent)

        # Pie Chart of Top 10
        top_contacts = df_logs["Phone"].value_counts().nlargest(10)
        fig, ax = plt.subplots()
        top_contacts.plot.pie(autopct='%1.0f%%', startangle=90, ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)

        # Table
        st.markdown("### Message Log")
        st.dataframe(df_logs.sort_values("Date Messaged", ascending=False))

        # Download
        csv = df_logs.to_csv(index=False).encode("utf-8")
        st.download_button("Download Full Log CSV", data=csv, file_name="message_log.csv")
