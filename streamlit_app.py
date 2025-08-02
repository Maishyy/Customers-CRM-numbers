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
import os
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')]
)
logger = logging.getLogger(__name__)

# --- Firebase Setup ---
TEST_MODE = st.sidebar.checkbox("Enable Test Mode", help="Use local Firebase emulator")

if TEST_MODE:
    os.environ["FIRESTORE_EMULATOR_HOST"] = "localhost:8080"
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate("local_test_creds.json"))
    db = firestore.client()
    st.sidebar.warning("TEST MODE ACTIVE - Using local emulator")
else:
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
PHONE_REGEX = r"(?:(?:\+?254|0)?(7[0-9]{2}|1[0-1][0-9])(?:[0-9]{6}))"
NAME_REGEX = r"(?:[A-Z][A-Z]+(?:\s+[A-Z][A-Z]+)+)(?=\s*(?:\d|[-+]|$))"
EXCLUDE_KEYWORDS = ["CDM", "BANK", "TRANSFER", "CHEQUE", "LTD", "LIMITED", "INTERNATIONAL"]
BLACKLISTED_NUMBERS = ["+254722000000", "+254000000000"]
MAX_FILE_SIZE_MB = 10
VALID_KENYAN_PREFIXES = ['70','71','72','73','74','75','76','77','78','79','11']

# --- Enhanced Helper Functions ---
def format_phone_number(phone):
    """Standardize phone to 254 format with rigorous validation"""
    if not phone:
        return None
        
    phone = str(phone).strip()
    phone = re.sub(r'[^\d]', '', phone)
    
    # Convert to 254 format
    if phone.startswith('0') and len(phone) == 10:
        phone = "254" + phone[1:]
    elif phone.startswith('7') and len(phone) == 9:
        phone = "254" + phone
    elif phone.startswith('254') and len(phone) == 12:
        pass  # Already in correct format
    else:
        return None
    
    # Validate Kenyan mobile number
    prefix = phone[3:5]
    if len(phone) != 12 or prefix not in VALID_KENYAN_PREFIXES:
        return None
    
    return f"+{phone}"

def clean_name(name):
    """Clean and standardize name formatting"""
    if not name or not isinstance(name, str):
        return ""
    
    # Remove unwanted suffixes/company indicators
    name = re.sub(r'\s*(TG[UV]|LTD|LIMITED|INTERNATIONAL|COMPANY)\s*$', '', name, flags=re.IGNORECASE)
    
    # Remove special characters except spaces and hyphens
    name = re.sub(r'[^\w\s-]', '', name)
    
    # Standardize capitalization (Title Case)
    name = ' '.join([part.capitalize() for part in name.split()])
    
    # Remove extra spaces
    name = ' '.join(name.split())
    
    return name.strip()

def should_exclude_line(line):
    """Check if line should be excluded from processing"""
    if not isinstance(line, str):
        return True
    
    line_upper = line.upper()
    return any(
        keyword in line_upper 
        for keyword in EXCLUDE_KEYWORDS
    ) or re.search(r'ACC(?:OUNT)?\s*NO|REF(?:ERENCE)?\s*NO', line_upper)

def validate_contact(phone, name):
    """Comprehensive contact validation"""
    if not phone or len(phone) != 13 or not phone.startswith("+254"):
        return False
    
    # Additional phone validation
    prefix = phone[4:6]
    if prefix not in VALID_KENYAN_PREFIXES:
        return False
    
    # Name validation (if present)
    if name:
        name_parts = name.split()
        if len(name_parts) < 1 or any(len(part) < 2 for part in name_parts):
            return False
            
    return True

# --- File Processors with Enhanced Quality Control ---
@lru_cache(maxsize=32)
def extract_contacts(text):
    """Extract contacts with improved validation and cleaning"""
    records = []
    seen_phones = set()
    
    for match in re.finditer(PHONE_REGEX, text):
        phone = format_phone_number(match.group())
        if not phone or phone in seen_phones:
            continue
        
        seen_phones.add(phone)
        context = text[max(0, match.start()-100):match.end()+100]
        
        # Extract and clean name
        name_match = re.search(NAME_REGEX, context)
        name = clean_name(name_match.group(0)) if name_match else ""
        
        # Only add if phone is valid
        if validate_contact(phone, name):
            records.append((phone, name))
    
    return records

def process_pdf(file):
    """Process PDF with duplicate prevention and progress tracking"""
    try:
        with pdfplumber.open(file) as pdf:
            total_pages = len(pdf.pages)
            progress_bar = st.progress(0)
            contacts = []
            seen_numbers = set()
            
            # Process every page (or every other for large files)
            pages = pdf.pages[::2] if total_pages > 20 else pdf.pages
            
            for i, page in enumerate(pages):
                if txt := page.extract_text():
                    for phone, name in extract_contacts(txt):
                        if phone and phone not in seen_numbers:
                            contacts.append((phone, name))
                            seen_numbers.add(phone)
                
                # Update progress
                progress_bar.progress((i + 1) / len(pages))
            
            return contacts
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        st.error(f"Error processing PDF: {str(e)}")
        return []

def process_csv(file):
    """Process CSV with chunking for large files"""
    try:
        if file.size > 5 * 1024 * 1024:  # 5MB
            chunks = pd.read_csv(file, chunksize=10000)
            return [
                contact 
                for chunk in chunks 
                for contact in extract_from_dataframe(chunk)
            ]
        return extract_from_dataframe(pd.read_csv(file))
    except Exception as e:
        logger.error(f"CSV processing error: {str(e)}")
        st.error(f"Error processing CSV: {str(e)}")
        return []

def process_excel(file):
    """Process Excel files with multiple sheets"""
    try:
        try:
            dfs = pd.read_excel(file, sheet_name=None, engine='openpyxl')
        except:
            dfs = pd.read_excel(file, sheet_name=None, engine='xlrd')
        
        # Limit to first 3 sheets if many exist
        sheets = list(dfs.items())[:3] if len(dfs) > 3 else dfs.items()
        return [
            contact
            for _, df in sheets
            for contact in extract_from_dataframe(df)
        ]
    except Exception as e:
        logger.error(f"Excel processing error: {str(e)}")
        st.error(f"Error processing Excel: {str(e)}")
        return []

def extract_from_dataframe(df):
    """Extract contacts from dataframe rows"""
    records = []
    for _, row in df.iterrows():
        line = " ".join(str(x) for x in row if pd.notna(x))
        if not should_exclude_line(line):
            records.extend(extract_contacts(line))
    return records

def process_file(file):
    """Main file processor with error handling"""
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
    """Process multiple files in parallel"""
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, files))
    return [contact for sublist in results for contact in sublist]

# --- Firebase Operations with Enhanced Validation ---
def save_to_firestore(data):
    """Save contacts to Firestore with duplicate prevention"""
    if not db or not data:
        return 0, 0
    
    collection = db.collection("contacts")
    batch = db.batch()
    new_count = 0
    duplicate_count = 0
    
    # First get existing numbers to minimize Firestore reads
    existing_numbers = set()
    docs = collection.select(["phone_number"]).stream()
    for doc in docs:
        existing_numbers.add(doc.get("phone_number"))
    
    # Process contacts
    for phone, name in {(p, n) for p, n in data if p}:
        if not validate_contact(phone, name):
            continue
            
        doc_ref = collection.document(phone.replace("+", ""))
        
        if phone in existing_numbers:
            duplicate_count += 1
            continue
            
        # Split name for better querying
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
            "timestamp": firestore.SERVER_TIMESTAMP,
            "last_transaction_date": firestore.SERVER_TIMESTAMP
        })
        new_count += 1
        existing_numbers.add(phone)
        
        # Commit in batches
        if new_count % 500 == 0:
            batch.commit()
            batch = db.batch()
    
    if new_count % 500 != 0:
        batch.commit()
    
    return new_count, duplicate_count

def log_message(phone, name):
    """Log sent messages with enhanced error handling"""
    try:
        db.collection("messages_sent").add({
            "phone_number": phone,
            "client_name": name,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "status": "queued"
        })
    except Exception as e:
        logger.error(f"Failed to log message for {phone}: {str(e)}")

@st.cache_data(ttl=3600)
def load_message_logs(days=30):
    """Load message logs with date filtering"""
    try:
        cutoff = datetime.now() - timedelta(days=days)
        docs = db.collection("messages_sent")\
                .where("timestamp", ">=", cutoff)\
                .stream()
        
        data = []
        for doc in docs:
            d = doc.to_dict()
            phone = d.get("phone_number", "")
            date = d.get("timestamp")
            date_str = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else ""
            data.append({
                "Phone": phone,
                "Date Messaged": date_str,
                "Status": d.get("status", "unknown")
            })
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading logs: {str(e)}")
        st.error("Could not load message logs.")
        return pd.DataFrame(columns=["Phone", "Date Messaged", "Status"])

# --- Enhanced Export Functions ---
def generate_standard_excel(data):
    """Generate standardized Excel output with quality checks"""
    formatted_data = []
    duplicate_check = set()
    
    for phone, name in data:
        # Skip duplicates in this export
        if phone in duplicate_check:
            continue
        duplicate_check.add(phone)
        
        # Format names
        first, last = "", ""
        if name:
            parts = name.strip().split()
            first = parts[0] if len(parts) > 0 else ""
            last = " ".join(parts[1:]) if len(parts) > 1 else ""
        
        clean_phone = phone.replace("+", "") if phone else ""
        
        formatted_data.append({
            "Firstname(optional)": first,
            "Lastname(optional)": last,
            "Phone or Email": clean_phone,
            "phone(254)": clean_phone,
            "Valid": "Yes" if validate_contact(phone, name) else "No"
        })
    
    df = pd.DataFrame(formatted_data)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
        # Add data validation
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        
        # Format header
        header_fmt = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#4472C4',
            'font_color': 'white',
            'border': 1
        })
        
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_fmt)
    
    buffer.seek(0)
    return buffer, df

def download_full_contact_list():
    """Download entire contact list with quality indicators"""
    try:
        contacts_ref = db.collection("contacts")
        docs = contacts_ref.stream()
        
        contact_list = []
        for doc in docs:
            data = doc.to_dict()
            contact_list.append((
                data.get("phone_number", ""),
                data.get("client_name", "")
            ))
        
        return generate_standard_excel(contact_list)
    except Exception as e:
        logger.error(f"Error downloading full contact list: {str(e)}")
        return None, None

# --- UI Components ---
st.set_page_config(layout="wide", page_title="Sequid Hardware Contact System")
tabs = st.tabs(["Upload & Sync", "Daily SMS List", "Dashboard", "Data Quality"])

# --- Tab 1: Upload & Sync ---
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
            with st.spinner("Processing files..."):
                all_data = process_files_parallel(uploaded_files) if len(uploaded_files) > 1 else process_file(uploaded_files[0])
            
            if all_data:
                # Preview data with quality indicators
                preview_df = pd.DataFrame(all_data, columns=["Phone", "Name"])
                preview_df["Valid"] = preview_df.apply(
                    lambda row: "✅" if validate_contact(row["Phone"], row["Name"]) else "❌", 
                    axis=1
                )
                
                st.subheader("Data Quality Preview")
                st.dataframe(preview_df.head(50))
                
                unique_count = len({p for p, _ in all_data if p})
                st.success(f"Found {unique_count} unique contacts")
                
                if st.button("Confirm Upload to Database"):
                    with st.spinner("Saving to database..."):
                        new_count, duplicate_count = save_to_firestore(all_data)
                    
                    st.success(f"""
                        - Added {new_count} new contacts
                        - Skipped {duplicate_count} duplicates
                    """)
                    
                    excel_file, df = generate_standard_excel(all_data)
                    st.download_button(
                        "Download Processed Contacts", 
                        data=excel_file, 
                        file_name="processed_contacts.xlsx",
                        mime="application/vnd.ms-excel"
                    )
            else:
                st.warning("No valid phone numbers found.")
    
    st.subheader("Contact Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Full Contact List"):
            with st.spinner("Preparing full contact list..."):
                excel_file, df = download_full_contact_list()
                if excel_file:
                    st.download_button(
                        label="Download Complete Contacts",
                        data=excel_file,
                        file_name="full_contact_list.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                    st.dataframe(df.head())
                else:
                    st.error("Failed to generate contact list")
    
    with col2:
        if st.button("Refresh Contact Count"):
            count = db.collection("contacts").count().get()[0][0].value
            st.metric("Total Contacts in Database", count)

# --- Tab 2: Daily SMS List ---
with tabs[1]:
    st.subheader("Generate Daily SMS List (Active customers only)")
    uploaded_file = st.file_uploader(
        "Upload today's statement", 
        type=["pdf", "csv", "xls", "xlsx"],
        key="daily_sms_uploader"
    )
    
    if uploaded_file and st.button("Generate Today's List"):
        current_data = process_file(uploaded_file)
        current_numbers = {p for p, _ in current_data if p}
        
        if not current_data:
            st.warning("No valid contacts found in this statement")
        else:
            # Get message history for these specific numbers
            sms_data = []
            batch_size = 300  # Firestore query limit
            current_batches = [
                list(current_numbers)[i:i+batch_size] 
                for i in range(0, len(current_numbers), batch_size)
            ]
            
            progress_bar = st.progress(0)
            total_batches = len(current_batches)
            
            for i, batch in enumerate(current_batches):
                # Get last message date for these numbers
                docs = db.collection("messages_sent")\
                        .where("phone_number", "in", batch)\
                        .order_by("timestamp", direction=firestore.Query.DESCENDING)\
                        .limit(1)\
                        .stream()
                
                last_messages = {doc.get("phone_number"): doc.get("timestamp") for doc in docs}
                
                # Check eligibility
                for phone, name in current_data:
                    if not phone or phone not in batch:
                        continue
                        
                    last_message = last_messages.get(phone)
                    eligible = (
                        not last_message or 
                        (datetime.now() - last_message).days > 7
                    )
                    
                    if eligible:
                        sms_data.append((phone, name))
                        log_message(phone, name)
                
                # Update progress
                progress_bar.progress((i + 1) / total_batches)
            
            if sms_data:
                sms_excel, sms_df = generate_standard_excel(sms_data)
                st.success(f"{len(sms_data)} active contacts ready for messaging.")
                st.download_button(
                    "Download SMS List", 
                    data=sms_excel, 
                    file_name=f"sms_list_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.ms-excel"
                )
                st.dataframe(sms_df)
            else:
                st.info("No eligible contacts to message today from this statement.")

# --- Tab 3: Dashboard ---
with tabs[2]:
    st.subheader("Message History Dashboard")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End date", datetime.now())
    
    if st.button("Refresh Dashboard"):
        df_logs = load_message_logs((end_date - start_date).days)
        
        if df_logs.empty:
            st.info("No message logs found for selected period.")
        else:
            # Metrics
            total_messages = len(df_logs)
            unique_contacts = df_logs["Phone"].nunique()
            success_rate = len(df_logs[df_logs["Status"] == "delivered"]) / total_messages if total_messages else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Messages", total_messages)
            col2.metric("Unique Contacts", unique_contacts)
            col3.metric("Success Rate", f"{success_rate:.1%}")
            
            # Visualizations
            st.subheader("Message Volume")
            daily_counts = df_logs.groupby("Date Messaged").size()
            st.bar_chart(daily_counts)
            
            st.subheader("Top Recipients")
            top_contacts = df_logs["Phone"].value_counts().nlargest(10)
            st.bar_chart(top_contacts)
            
            # Raw data
            st.subheader("Message Log")
            st.dataframe(df_logs.sort_values("Date Messaged", ascending=False))
            
            # Export
            csv = df_logs.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Full Log CSV", 
                data=csv, 
                file_name="message_log.csv",
                mime="text/csv"
            )

# --- Tab 4: Data Quality ---
with tabs[3]:
    st.subheader("Data Quality Analysis")
    
    if st.button("Run Quality Check"):
        with st.spinner("Analyzing database..."):
            # Get all contacts
            contacts_ref = db.collection("contacts")
            docs = contacts_ref.stream()
            
            # Collect stats
            stats = {
                "total_contacts": 0,
                "invalid_phones": 0,
                "missing_names": 0,
                "duplicate_phones": defaultdict(int),
                "phone_prefixes": defaultdict(int)
            }
            
            phone_counts = defaultdict(int)
            
            for doc in docs:
                data = doc.to_dict()
                phone = data.get("phone_number", "")
                name = data.get("client_name", "")
                
                stats["total_contacts"] += 1
                phone_counts[phone] += 1
                
                # Check phone validity
                if not validate_contact(phone, name):
                    stats["invalid_phones"] += 1
                
                # Check name presence
                if not name.strip():
                    stats["missing_names"] += 1
                
                # Track phone prefixes
                if phone.startswith("+254") and len(phone) >= 6:
                    prefix = phone[4:6]
                    stats["phone_prefixes"][prefix] += 1
            
            # Find duplicates
            stats["duplicate_count"] = sum(1 for count in phone_counts.values() if count > 1)
            
            # Display results
            st.metric("Total Contacts", stats["total_contacts"])
            
            col1, col2 = st.columns(2)
            col1.metric("Invalid Phone Numbers", stats["invalid_phones"])
            col2.metric("Contacts Missing Names", stats["missing_names"])
            
            st.metric("Duplicate Phone Numbers", stats["duplicate_count"])
            
            # Show prefix distribution
            st.subheader("Phone Prefix Distribution")
            prefix_df = pd.DataFrame.from_dict(
                stats["phone_prefixes"], 
                orient="index", 
                columns=["Count"]
            ).sort_values("Count", ascending=False)
            st.bar_chart(prefix_df)
            
            # Show sample duplicates
            if stats["duplicate_count"] > 0:
                st.subheader("Sample Duplicate Entries")
                duplicates = {p: c for p, c in phone_counts.items() if c > 1}
                sample_dupes = list(duplicates.items())[:10]
                
                for phone, count in sample_dupes:
                    st.write(f"{phone}: {count} occurrences")
                    docs = contacts_ref.where("phone_number", "==", phone).stream()
                    for doc in docs:
                        st.write(f"- {doc.to_dict().get('client_name', 'No name')}")