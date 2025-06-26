import re
import pandas as pd
import streamlit as st
import pdfplumber
from io import BytesIO
import firebase_admin
from firebase_admin import credentials, firestore
import json
from datetime import datetime, timedelta
import altair as alt
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')]
)
logger = logging.getLogger(__name__)

# Constants
PHONE_REGEX = r"(?:(?:\+254|254|0)?(7\d{8}|11\d{7}))"
NAME_REGEX = r"([A-Z][A-Z]+\s+){1,}[A-Z][A-Z]+"
EXCLUDE_KEYWORDS = ["CDM", "BANK", "TRANSFER", "CHEQUE"]
BLACKLISTED_NUMBERS = ["+254722000000", "+254000000000"]
MAX_FILE_SIZE_MB = 10

# Initialize Firebase
def init_firebase():
    try:
        if not firebase_admin._apps:
            if 'firebase_creds' in st.secrets:
                cred = credentials.Certificate(json.loads(st.secrets["firebase_creds"]))
            else:
                cred = credentials.Certificate('firebase-creds.json')
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        logger.error(f"Firebase init error: {str(e)}")
        st.error("Firebase initialization failed")
        return None

db = init_firebase()

# Core Phone Processing Functions
def normalize_phone(raw):
    if not raw: return None
    try:
        digits = re.sub(r'[^\d]', '', raw)
        if len(digits) == 9 and digits.startswith('7'):
            normalized = f"+254{digits}"
        elif len(digits) == 12 and digits.startswith('254'):
            normalized = f"+{digits}"
        elif len(digits) == 10 and digits.startswith('0'):
            normalized = f"+254{digits[1:]}"
        else:
            return None
            
        return normalized if normalized not in BLACKLISTED_NUMBERS and normalized.startswith("+2547") else None
    except Exception:
        return None

def should_exclude(text):
    return any(kw in text.upper() for kw in EXCLUDE_KEYWORDS) if isinstance(text, str) else True

def extract_contacts(text):
    records = []
    for match in re.finditer(PHONE_REGEX, text):
        phone = normalize_phone(match.group())
        if not phone: continue
        
        after_num = text[match.end():].strip()
        name_match = re.search(NAME_REGEX, after_num)
        name = name_match.group(0).strip() if name_match else ""
        
        records.append((phone, name))
    return records

# File Processing
def process_pdf(file):
    try:
        with pdfplumber.open(file) as pdf:
            return [contact for page in pdf.pages 
                    if (text := page.extract_text()) 
                    for contact in extract_contacts(text)]
    except Exception as e:
        logger.error(f"PDF error: {str(e)}")
        st.error(f"PDF processing failed: {e}")
        return []

def process_csv(file):
    try:
        df = pd.read_csv(file)
        return [(normalize_phone(str(num)), name) 
                for num, name in df.values 
                if not should_exclude(str(num))]
    except Exception as e:
        logger.error(f"CSV error: {str(e)}")
        st.error(f"CSV processing failed: {e}")
        return []

def process_excel(file):
    try:
        dfs = pd.read_excel(file, sheet_name=None, engine='openpyxl')
        return [contact for df in dfs.values() 
                for contact in extract_from_dataframe(df)]
    except Exception as e:
        logger.error(f"Excel error: {str(e)}")
        st.error(f"Excel processing failed: {e}")
        return []

def extract_from_dataframe(df):
    return [contact for _, row in df.iterrows() 
            if not should_exclude(str(row))
            for contact in extract_contacts(" ".join(str(x) for x in row if pd.notna(x)))]

# Firebase Operations
def save_contacts(contacts, dry_run=False):
    if not db: return [], 0, 0, 0
    
    collection = db.collection("contacts")
    results = []
    new = dup = err = 0
    
    for phone, name in contacts:
        if not phone: continue
        
        try:
            doc_ref = collection.document(phone.replace("+", ""))
            exists = doc_ref.get().exists
            
            if dry_run:
                results.append({
                    "Phone": phone,
                    "Name": name,
                    "Status": "Duplicate" if exists else "New"
                })
            elif not exists:
                first, last = (name.split(" ", 1) + [""])[:2]
                doc_ref.set({
                    "phone_number": phone,
                    "client_name": name,
                    "first_name": first,
                    "last_name": last,
                    "timestamp": firestore.SERVER_TIMESTAMP
                })
            
            dup += exists
            new += not exists
        except Exception as e:
            err += 1
            logger.error(f"Save error: {phone} - {str(e)}")
            if dry_run:
                results[-1]["Status"] = "Error"
                results[-1]["Error"] = str(e)
    
    return (results, new, dup, err) if dry_run else (new, dup, err)

# Streamlit UI
def main():
    st.set_page_config(layout="wide", page_title="Contact Manager")
    
    tabs = st.tabs(["Upload", "SMS List", "Dashboard"])
    
    with tabs[0]:
        st.header("Upload Contacts")
        dry_run = st.checkbox("Dry Run Mode", True)
        
        files = st.file_uploader("Upload files", 
                               type=["pdf", "csv", "xlsx"], 
                               accept_multiple_files=True)
        
        if files:
            contacts = []
            for file in files:
                ext = file.name.split(".")[-1].lower()
                if ext == "pdf":
                    contacts.extend(process_pdf(file))
                elif ext == "csv":
                    contacts.extend(process_csv(file))
                elif ext in ("xls", "xlsx"):
                    contacts.extend(process_excel(file))
            
            if contacts:
                if dry_run:
                    results, new, dup, err = save_contacts(contacts, True)
                    st.dataframe(pd.DataFrame(results).head(50))
                else:
                    new, dup, err = save_contacts(contacts)
                
                st.success(f"Processed {len(contacts)} contacts")
                st.download_button(
                    "Download Results",
                    pd.DataFrame(contacts).to_csv().encode(),
                    "contacts.csv"
                )
    
    with tabs[1]:
        st.header("SMS List Generator")
        if st.button("Generate List"):
            # Implement SMS list generation
            pass
    
    with tabs[2]:
        st.header("Dashboard")
        # Implement dashboard
        pass

if __name__ == "__main__":
    main()