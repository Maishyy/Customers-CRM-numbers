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
MAX_FILE_SIZE_MB = 10

# --- Normalize Numbers ---
def normalize_number(raw):
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
    if not isinstance(line, str):
        return True
    return any(keyword in line.upper() for keyword in EXCLUDE_KEYWORDS)

# --- File Processors ---
def process_file(file):
    try:
        ext = file.name.lower().split('.')[-1]
        if ext == "pdf":
            return process_pdf(file)
        elif ext == "csv":
            return process_csv(file)
        elif ext in ("xls", "xlsx"):
            return process_excel(file)
        else:
            st.warning(f"Unsupported file format: {file.name}")
            return []
    except Exception as e:
        logger.error(f"Error processing {file.name}: {str(e)}")
        st.error(f"Error processing {file.name}: {str(e)}")
        return []

def process_files_parallel(files):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, files))
    return [c for sub in results for c in sub]

@lru_cache(maxsize=32)
def extract_contacts(text):
    records = []
    for match in re.finditer(PHONE_REGEX, text):
        phone = normalize_number(match.group())
        if not phone:
            continue
        after = text[match.end():].strip()
        name_match = re.search(NAME_REGEX, after)
        name = name_match.group(0).strip() if name_match else ""
        records.append((phone, name))
    return records

def process_pdf(file):
    try:
        with pdfplumber.open(file) as pdf:
            pages = pdf.pages[::2] if len(pdf.pages) > 20 else pdf.pages
            return [c for page in pages if (txt := page.extract_text()) for c in extract_contacts(txt)]
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        return []

def process_csv(file):
    try:
        if file.size > 5 * 1024 * 1024:
            chunks = pd.read_csv(file, chunksize=10000)
            return [c for chunk in chunks for c in extract_from_dataframe(chunk)]
        return extract_from_dataframe(pd.read_csv(file))
    except Exception as e:
        logger.error(f"CSV processing error: {str(e)}")
        return []

def process_excel(file):
    try:
        try:
            dfs = pd.read_excel(file, sheet_name=None, engine='openpyxl')
        except:
            dfs = pd.read_excel(file, sheet_name=None, engine='xlrd')
        sheets = list(dfs.items())[:3] if len(dfs) > 3 else dfs.items()
        return [c for _, df in sheets for c in extract_from_dataframe(df)]
    except Exception as e:
        logger.error(f"Excel processing error: {str(e)}")
        return []

def extract_from_dataframe(df):
    records = []
    for _, row in df.iterrows():
        line = " ".join(str(x) for x in row if pd.notna(x))
        if not should_exclude_line(line):
            records.extend(extract_contacts(line))
    return records

# --- Firebase Save ---
def save_to_firestore(data):
    if not db or not data:
        return 0
    collection = db.collection("contacts")
    batch = db.batch()
    new_count = 0
    unique = {(p, n) for p, n in data if p}
    for phone, name in unique:
        doc_ref = collection.document(phone.replace("+", ""))
        if not doc_ref.get().exists:
            first, last = "", ""
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
            if new_count % 500 == 0:
                batch.commit()
                batch = db.batch()
    if new_count % 500 != 0:
        batch.commit()
    return new_count

# --- Load Message Logs ---
@st.cache_data
def load_message_logs():
    try:
        docs = db.collection("messages_sent").stream()
        data = []
        for doc in docs:
            d = doc.to_dict()
            phone = d.get("phone_number", "")
            date = d.get("timestamp")
            date_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else ""
            data.append({"Phone": phone, "Date Messaged": date_str})
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading logs: {str(e)}")
        st.error("Could not load message logs.")
        return pd.DataFrame(columns=["Phone", "Date Messaged"])

# --- Log Sent Message ---
def log_message(phone, name):
    try:
        db.collection("messages_sent").add({
            "phone_number": phone,
            "client_name": name,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        logger.error(f"Log error for {phone}: {str(e)}")

# --- Export Helper ---
def generate_excel(data):
    df = pd.DataFrame(data, columns=["Phone Number", "Client Name"])
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer, df

# --- UI ---
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
        oversized = [f.name for f in uploaded_files if f.size > MAX_FILE_SIZE_MB * 1024 * 1024]
        if oversized:
            st.error(f"Oversized files: {', '.join(oversized)}")
        else:
            with st.spinner("Processing..."):
                all_data = process_files_parallel(uploaded_files) if len(uploaded_files) > 1 else process_file(uploaded_files[0])
            if all_data:
                unique_count = len({p for p, _ in all_data if p})
                st.success(f"Found {unique_count} unique contacts")
                new_count = save_to_firestore(all_data)
                st.success(f"Added {new_count} new contacts")
                excel_file, df = generate_excel(all_data)
                st.download_button("Download Excel", data=excel_file, file_name="contacts.xlsx")
                st.dataframe(df.head())
            else:
                st.warning("No valid phone numbers found.")

# --- Tab 2: Daily SMS List ---
with tabs[1]:
    st.subheader("Generate Daily SMS List (only message once per 7 days)")
    if st.button("Generate Today's List"):
        all_contacts = db.collection("contacts").stream()
        sent = load_message_logs()
        recent_phones = sent[sent["Date Messaged"] >= (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")]["Phone"].tolist()

        sms_data = []
        for doc in all_contacts:
            d = doc.to_dict()
            phone = d.get("phone_number", "")
            name = d.get("client_name", "")
            if phone and phone not in recent_phones:
                sms_data.append((phone, name))
                log_message(phone, name)  # log that message is sent

        if sms_data:
            sms_excel, sms_df = generate_excel(sms_data)
            st.success(f"{len(sms_data)} new contacts ready for messaging.")
            st.download_button("Download SMS List", data=sms_excel, file_name="todays_sms_list.xlsx")
            st.dataframe(sms_df)
        else:
            st.info("No new contacts to message today.")

# --- Tab 3: Dashboard ---
with tabs[2]:
    st.subheader("Message History Dashboard")
    df_logs = load_message_logs()
    if df_logs.empty:
        st.info("No message logs found.")
    else:
        st.metric("Total Messages Sent", len(df_logs))
        recent = df_logs["Date Messaged"].value_counts().sort_index()
        st.bar_chart(recent)
        top_contacts = df_logs["Phone"].value_counts().nlargest(10)
        fig, ax = plt.subplots()
        top_contacts.plot.pie(autopct='%1.0f%%', startangle=90, ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)
        st.markdown("### Message Log")
        st.dataframe(df_logs.sort_values("Date Messaged", ascending=False))
        csv = df_logs.to_csv(index=False).encode("utf-8")
        st.download_button("Download Full Log CSV", data=csv, file_name="message_log.csv")
