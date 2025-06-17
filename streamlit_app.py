import re
import pandas as pd
import streamlit as st
import pdfplumber
from io import BytesIO
import firebase_admin
from firebase_admin import credentials, firestore
import json

# --- Firebase Initialization ---
firebase_json = json.loads(st.secrets["firebase_creds"])
cred = credentials.Certificate(firebase_json)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Constants ---
PHONE_REGEX = r"(?:(?:\+254|254|0)?(7\d{8}|11\d{7}))"
NAME_REGEX = r"([A-Z][A-Z]+\s+){1,}[A-Z][A-Z]+"
EXCLUDE_KEYWORDS = ["CDM", "BANK", "TRANSFER", "CHEQUE"]
BLACKLISTED_NUMBERS = ["+254722000000", "+254000000000"]

# --- Normalization & Filters ---
def normalize_number(raw):
    if not raw: return None
    normalized = f"+254{raw[-9:]}"
    if normalized in BLACKLISTED_NUMBERS or not normalized.startswith("+2547"):
        return None
    return normalized

def should_exclude_line(line):
    return any(keyword in line.upper() for keyword in EXCLUDE_KEYWORDS)

# --- Extraction Logic ---
def extract_from_text_lines(text_lines):
    records = []
    for line in text_lines:
        if should_exclude_line(line): continue
        matches = re.findall(PHONE_REGEX, line)
        for match in matches:
            phone = normalize_number(match)
            if not phone: continue
            after_number = re.split(match, line, maxsplit=1)[-1]
            name_match = re.search(NAME_REGEX, after_number)
            name = name_match.group(0).strip() if name_match else ""
            records.append((phone, name))
    return records

def extract_from_dataframe(df):
    records = []
    for _, row in df.iterrows():
        row_text = " ".join(row.astype(str))
        if should_exclude_line(row_text): continue
        matches = re.findall(PHONE_REGEX, row_text)
        for match in matches:
            phone = normalize_number(match)
            if not phone: continue
            after_number = re.split(match, row_text, maxsplit=1)[-1]
            name_match = re.search(NAME_REGEX, after_number)
            name = name_match.group(0).strip() if name_match else ""
            records.append((phone, name))
    return records

def process_pdf(file):
    records = []
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                lines = page.extract_text().split("\n")
                records.extend(extract_from_text_lines(lines))
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return records

def process_csv(file):
    try:
        df = pd.read_csv(file)
        return extract_from_dataframe(df)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return []

def process_excel(file):
    try:
        try:
            dfs = pd.read_excel(file, sheet_name=None, engine='openpyxl')
        except Exception:
            dfs = pd.read_excel(file, sheet_name=None, engine='xlrd')

        all_records = []
        for df in dfs.values():
            all_records.extend(extract_from_dataframe(df))
        return all_records
    except Exception as e:
        st.error(f"Error reading Excel: {e}")
        return []

# --- Firebase Save Logic ---
def save_to_firestore(data):
    collection = db.collection("contacts")
    new_count = 0

    for phone, name in data:
        if not phone: continue
        doc_id = phone.replace("+", "")
        doc_ref = collection.document(doc_id)

        if not doc_ref.get().exists:
            first, last = "", ""
            if name:
                split_name = name.strip().split(" ", 1)
                first = split_name[0]
                last = split_name[1] if len(split_name) > 1 else ""

            doc_ref.set({
                "phone_number": phone,
                "client_name": name,
                "first_name": first,
                "last_name": last,
                "source": "upload",
                "timestamp": firestore.SERVER_TIMESTAMP
            })
            new_count += 1
    return new_count

# --- Excel Output ---
def generate_excel(data):
    unique = {(p, n) for p, n in data if p}
    df = pd.DataFrame(sorted(unique), columns=["Phone Number", "Client Name"])
    df[["First Name", "Last Name"]] = df["Client Name"].str.split(n=1, expand=True)
    df["Phone (Raw)"] = df["Phone Number"].str.replace("+254", "")
    df["Phone (254xxxxxxxxx)"] = df["Phone Number"].str.replace("+", "")
    final_df = pd.DataFrame()
    final_df["Firstname(optional)"] = df["First Name"]
    final_df["Lastname(optional)"] = df["Last Name"]
    final_df["Phone or Email"] = df["Phone Number"]
    final_df["Phone (Raw)"] = df["Phone (Raw)"]
    final_df["Phone (254 format)"] = df["Phone (254xxxxxxxxx)"]
    final_df["Client Name (Original)"] = df["Client Name"]
    output = BytesIO()
    final_df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    return output

# --- Streamlit UI ---
st.title("Extract + Sync to Firebase")
st.write("Upload MPESA or bank statements (PDF, Excel, CSV).")
st.write("Filtered contacts will be saved to Firebase and downloadable as Excel.")

uploaded_files = st.file_uploader("Upload files", type=["pdf", "csv", "xls", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        filename = file.name.lower()
        st.write(f"Processing: {file.name}")
        if filename.endswith(".pdf"):
            all_data.extend(process_pdf(file))
        elif filename.endswith(".csv"):
            all_data.extend(process_csv(file))
        elif filename.endswith((".xls", ".xlsx")):
            all_data.extend(process_excel(file))
        else:
            st.warning(f"Unsupported file format: {file.name}")

    if all_data:
        new_count = save_to_firestore(all_data)
        st.success(f"Synced {len(set(all_data))} contacts. {new_count} new added.")
        excel_file = generate_excel(all_data)
        st.download_button("Download Excel", data=excel_file, file_name="contacts.xlsx")
        st.write("Preview of uploaded data:")
        st.dataframe(pd.read_excel(excel_file).head())
    else:
        st.warning("No valid phone numbers found.")
