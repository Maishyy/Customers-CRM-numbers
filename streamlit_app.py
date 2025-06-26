# streamlit_app.py
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

# --- Firebase Setup ---
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

# --- Helpers ---
def normalize_number(raw):
    if not raw: return None
    normalized = f"+254{raw[-9:]}"
    if normalized in BLACKLISTED_NUMBERS or not normalized.startswith("+2547"):
        return None
    return normalized

def should_exclude_line(line):
    return any(keyword in line.upper() for keyword in EXCLUDE_KEYWORDS)

# --- Extractors ---
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
        if file.name.endswith(".xls"):
            dfs = pd.read_excel(file, sheet_name=None, engine="xlrd")
        else:
            dfs = pd.read_excel(file, sheet_name=None, engine="openpyxl")
        all_records = []
        for df in dfs.values():
            all_records.extend(extract_from_dataframe(df))
        return all_records
    except Exception as e:
        st.error(f"Error reading Excel file '{file.name}': {e}")
        return []

# --- Firebase Logic ---
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

def log_message_sent(phone, name):
    today_str = datetime.now().strftime("%Y-%m-%d")
    doc_id = f"{phone.replace('+', '')}_{today_str}"
    db.collection("messages_sent").document(doc_id).set({
        "phone_number": phone,
        "name": name,
        "date": today_str,
        "timestamp": firestore.SERVER_TIMESTAMP
    })

def was_messaged_within_last_7_days(phone):
    cutoff_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
    messages = db.collection("messages_sent")\
        .where("phone_number", "==", phone)\
        .where("date", ">=", cutoff_date)\
        .limit(1).stream()
    return any(True for _ in messages)

def get_eligible_sms_contacts():
    contacts_ref = db.collection("contacts").stream()
    to_message = []

    for doc in contacts_ref:
        data = doc.to_dict()
        phone = data.get("phone_number")
        name = data.get("client_name", "")
        if phone and not was_messaged_within_last_7_days(phone):
            to_message.append((phone, name))
            log_message_sent(phone, name)
    return to_message

def generate_excel(data):
    unique = {(p, n) for p, n in data if p}
    df = pd.DataFrame(sorted(unique), columns=["Phone Number", "Client Name"])
    df[["First Name", "Last Name"]] = df["Client Name"].str.split(n=1, expand=True)
    df["Phone (Raw)"] = df["Phone Number"].str.replace("+254", "")
    df["Phone (254xxxxxxxxx)"] = df["Phone Number"].str.replace("+", "")
    output = BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)
    return output, df

def load_message_logs():
    docs = db.collection("messages_sent").stream()
    records = []
    for doc in docs:
        data = doc.to_dict()
        records.append({
            "Phone": data.get("phone_number", ""),
            "Name": data.get("name", ""),
            "Date Messaged": data.get("date", ""),
        })
    return pd.DataFrame(records)

# --- Streamlit UI ---
st.title("Sequid Hardware Contact System")

tabs = st.tabs(["Upload & Sync", "Daily SMS List", "Dashboard"])

# --- Tab 1: Upload ---
with tabs[0]:
    st.subheader("Upload MPESA or Bank Statements")
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
            excel_file, df = generate_excel(all_data)
            st.download_button("Download Excel", data=excel_file, file_name="contacts.xlsx")
            st.dataframe(df.head())
        else:
            st.warning("No valid phone numbers found.")

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
