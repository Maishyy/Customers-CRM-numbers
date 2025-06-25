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
from tqdm import tqdm
from collections import defaultdict

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# --- Constants ---
PHONE_REGEX = r"(?:(?:\+254|254|0)?(7\d{8}|11\d{7}))"
NAME_REGEX = r"([A-Za-z][A-Za-z]+\s+){1,}[A-Za-z][A-Za-z]+"  # More inclusive name matching
EXCLUDE_KEYWORDS = ["CDM", "BANK", "TRANSFER", "CHEQUE", "DEPOSIT"]
BLACKLISTED_NUMBERS = ["+254722000000", "+254000000000"]
DRY_RUN_MAX_DISPLAY = 100  # Max records to display in dry run
MAX_FILE_SIZE_MB = 10  # Maximum file size allowed

# --- Firebase Initialization ---
def initialize_firebase():
    try:
        if not firebase_admin._apps:
            if 'firebase_creds' in st.secrets:
                firebase_json = json.loads(st.secrets["firebase_creds"])
                cred = credentials.Certificate(firebase_json)
                firebase_admin.initialize_app(cred)
            else:
                # For local testing without secrets.toml
                cred = credentials.Certificate('firebase-creds.json')
                firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        logger.error(f"Firebase initialization failed: {str(e)}")
        st.error("Failed to initialize Firebase. Check logs for details.")
        return None

db = initialize_firebase()

# --- Utility Functions ---
def normalize_number(raw):
    """Normalize phone numbers to +254 format"""
    if not raw: 
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
    except:
        return None

def should_exclude_line(line):
    """Check if line contains excluded keywords"""
    if not isinstance(line, str):
        return False
    return any(keyword in line.upper() for keyword in EXCLUDE_KEYWORDS)

# --- Data Processing Functions ---
def extract_from_text_lines(text_lines):
    """Extract contacts from text lines"""
    records = []
    for line in text_lines:
        if not line or should_exclude_line(line): 
            continue
            
        # Find all phone matches in the line
        matches = re.finditer(PHONE_REGEX, line)
        for match in matches:
            phone = normalize_number(match.group())
            if not phone: 
                continue
                
            # Extract name after the phone number
            after_number = line[match.end():].strip()
            name_match = re.search(NAME_REGEX, after_number)
            name = name_match.group(0).strip() if name_match else ""
            
            records.append((phone, name))
    return records

def process_pdf(file):
    """Process PDF file with progress tracking"""
    records = []
    try:
        with pdfplumber.open(file) as pdf:
            with st.spinner(f"Processing {file.name}..."):
                progress_bar = st.progress(0)
                total_pages = len(pdf.pages)
                
                for i, page in enumerate(tqdm(pdf.pages, desc="Processing PDF")):
                    text = page.extract_text()
                    if text:
                        records.extend(extract_from_text_lines(text.split("\n")))
                    progress_bar.progress((i + 1) / total_pages)
                
                progress_bar.empty()
        logger.info(f"Processed PDF: {file.name}, found {len(records)} contacts")
    except Exception as e:
        logger.error(f"Error processing PDF {file.name}: {str(e)}")
        st.error(f"Error reading PDF: {e}")
    return records

def process_csv(file):
    """Process CSV file with chunking for large files"""
    records = []
    try:
        with st.spinner(f"Processing {file.name}..."):
            # Try to determine file size for chunking
            file_size = file.size / (1024 * 1024)  # MB
            chunksize = 10**5 if file_size > 5 else None  # Use chunks for files >5MB
            
            if chunksize:
                for chunk in tqdm(pd.read_csv(file, chunksize=chunksize), desc="Processing CSV"):
                    records.extend(extract_from_dataframe(chunk))
            else:
                df = pd.read_csv(file)
                records.extend(extract_from_dataframe(df))
                
        logger.info(f"Processed CSV: {file.name}, found {len(records)} contacts")
    except Exception as e:
        logger.error(f"Error processing CSV {file.name}: {str(e)}")
        st.error(f"Error reading CSV: {e}")
    return records

def process_excel(file):
    """Process Excel file with multiple sheets"""
    records = []
    try:
        with st.spinner(f"Processing {file.name}..."):
            try:
                dfs = pd.read_excel(file, sheet_name=None, engine='openpyxl')
            except Exception:
                dfs = pd.read_excel(file, sheet_name=None, engine='xlrd')
                
            for sheet_name, df in dfs.items():
                records.extend(extract_from_dataframe(df))
                
        logger.info(f"Processed Excel: {file.name}, found {len(records)} contacts")
    except Exception as e:
        logger.error(f"Error processing Excel {file.name}: {str(e)}")
        st.error(f"Error reading Excel: {e}")
    return records

def extract_from_dataframe(df):
    """Extract contacts from DataFrame"""
    records = []
    for _, row in df.iterrows():
        row_text = " ".join(str(x) for x in row if pd.notna(x))
        if should_exclude_line(row_text): 
            continue
            
        matches = re.finditer(PHONE_REGEX, row_text)
        for match in matches:
            phone = normalize_number(match.group())
            if not phone: 
                continue
                
            after_number = row_text[match.end():].strip()
            name_match = re.search(NAME_REGEX, after_number)
            name = name_match.group(0).strip() if name_match else ""
            
            records.append((phone, name))
    return records

# --- Firebase Operations ---
def save_to_firestore(data, dry_run=False):
    """Save contacts to Firestore with dry run option"""
    if not db:
        st.error("Database not initialized")
        return [], 0, 0, 0 if dry_run else (0, 0, 0)
        
    collection = db.collection("contacts")
    new_count = 0
    duplicate_count = 0
    error_count = 0
    dry_run_results = []
    
    # Deduplicate within current batch
    batch_seen = set()
    unique_data = []
    
    for phone, name in data:
        if not phone:
            continue
        if phone in batch_seen:
            duplicate_count += 1
            continue
        batch_seen.add(phone)
        unique_data.append((phone, name))
    
    # Process records
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(unique_data)
    
    for i, (phone, name) in enumerate(unique_data):
        try:
            doc_id = phone.replace("+", "")
            doc_ref = collection.document(doc_id)
            
            progress = (i + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Processing {i+1}/{total} - {phone}")
            
            if dry_run:
                # Dry run - check existence only
                exists = doc_ref.get().exists
                dry_run_results.append({
                    "Phone": phone,
                    "Name": name,
                    "Status": "Duplicate" if exists else "New",
                    "Exists": exists
                })
                if exists:
                    duplicate_count += 1
                else:
                    new_count += 1
            else:
                # Actual save operation
                if not doc_ref.get().exists:
                    first, last = "", ""
                    if name:
                        parts = name.strip().split(" ", 1)
                        first = parts[0]
                        last = parts[1] if len(parts) > 1 else ""

                    doc_ref.set({
                        "phone_number": phone,
                        "client_name": name,
                        "first_name": first,
                        "last_name": last,
                        "source": "upload",
                        "timestamp": firestore.SERVER_TIMESTAMP
                    })
                    new_count += 1
                    logger.info(f"Added new contact: {phone}")
                else:
                    duplicate_count += 1
                    logger.debug(f"Duplicate skipped: {phone}")
                    
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing {phone}: {str(e)}")
            if dry_run:
                dry_run_results[-1]["Status"] = "Error"
                dry_run_results[-1]["Error"] = str(e)
    
    progress_bar.empty()
    status_text.empty()
    
    logger.info(
        f"{'Dry run' if dry_run else 'Upload'} results - "
        f"New: {new_count}, Duplicates: {duplicate_count}, Errors: {error_count}"
    )
    
    if dry_run:
        return dry_run_results, new_count, duplicate_count, error_count
    return new_count, duplicate_count, error_count

def was_messaged_recently(phone, days=7):
    """Check if phone was messaged within given days"""
    if not db:
        return False
        
    try:
        messages_ref = db.collection("messages_sent")\
                         .where("phone_number", "==", phone)\
                         .order_by("timestamp", direction=firestore.Query.DESCENDING)\
                         .limit(1).stream()

        for doc in messages_ref:
            data = doc.to_dict()
            last_ts = data.get("timestamp")
            if last_ts:
                last_time = last_ts.replace(tzinfo=None)
                return datetime.utcnow() - last_time < timedelta(days=days)
        return False
    except Exception as e:
        logger.error(f"Error checking message history for {phone}: {str(e)}")
        return False

def log_message_sent(phone, name):
    """Log sent message to Firestore"""
    if not db:
        return
        
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        doc_id = f"{phone.replace('+', '')}_{today_str}"
        db.collection("messages_sent").document(doc_id).set({
            "phone_number": phone,
            "name": name,
            "date": today_str,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        logger.info(f"Logged message to {phone}")
    except Exception as e:
        logger.error(f"Error logging message for {phone}: {str(e)}")

# --- Reporting Functions ---
def generate_sms_excel(data):
    """Generate Excel file with contacts"""
    df = pd.DataFrame(data, columns=["Phone Number", "Client Name"])
    df["Date Messaged"] = datetime.now().strftime("%Y-%m-%d")
    df["Phone (No Plus)"] = "254" + df["Phone Number"].str[-9:]
    output = BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    return output, df

def load_dashboard_data(days=30):
    """Load data for dashboard"""
    if not db:
        return pd.DataFrame()
        
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        docs = db.collection("messages_sent")\
                 .where("timestamp", ">=", cutoff_date)\
                 .stream()
        
        rows = []
        for doc in docs:
            data = doc.to_dict()
            rows.append({
                "phone": data.get("phone_number"),
                "name": data.get("name"),
                "date": data.get("date"),
                "timestamp": data.get("timestamp")
            })
        return pd.DataFrame(rows)
    except Exception as e:
        logger.error(f"Error loading dashboard data: {str(e)}")
        return pd.DataFrame()

def get_eligible_sms_contacts():
    """Get contacts eligible for messaging"""
    if not db:
        return []
        
    try:
        contacts_ref = db.collection("contacts").stream()
        to_message = []
        
        for doc in contacts_ref:
            data = doc.to_dict()
            phone = data.get("phone_number")
            name = data.get("client_name", "")
            
            if phone and not was_messaged_recently(phone):
                to_message.append((phone, name))
                log_message_sent(phone, name)
                
        return to_message
    except Exception as e:
        logger.error(f"Error getting SMS contacts: {str(e)}")
        return []

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Contact Management System")

# Initialize session state for dry run
if 'dry_run_mode' not in st.session_state:
    st.session_state.dry_run_mode = True

# --- Tabs ---
tabs = st.tabs(["Upload & Sync", "Daily SMS List", "Dashboard"])

with tabs[0]:
    st.header("Upload & Sync Contacts")
    
    # Dry run toggle
    col1, col2 = st.columns(2)
    with col1:
        dry_run = st.checkbox(
            "Dry Run Mode", 
            value=st.session_state.dry_run_mode,
            help="Preview changes without writing to database"
        )
        st.session_state.dry_run_mode = dry_run
    with col2:
        if dry_run:
            st.warning("DRY RUN MODE ACTIVE - No changes will be saved")
    
    # File uploader with size limit
    uploaded_files = st.file_uploader(
        "Upload files (PDF, CSV, Excel)", 
        type=["pdf", "csv", "xls", "xlsx"], 
        accept_multiple_files=True,
        help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB each"
    )
    
    if uploaded_files:
        # Check file sizes
        oversized_files = [
            f.name for f in uploaded_files 
            if f.size > MAX_FILE_SIZE_MB * 1024 * 1024
        ]
        
        if oversized_files:
            st.error(f"Files exceed size limit ({MAX_FILE_SIZE_MB}MB): {', '.join(oversized_files)}")
        else:
            all_data = []
            total_files = len(uploaded_files)
            
            with st.expander("Processing Progress", expanded=True):
                file_progress = st.progress(0)
                file_status = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    file_status.text(f"Processing {i+1}/{total_files}: {file.name}")
                    
                    try:
                        if file.name.lower().endswith(".pdf"):
                            all_data.extend(process_pdf(file))
                        elif file.name.lower().endswith(".csv"):
                            all_data.extend(process_csv(file))
                        elif file.name.lower().endswith((".xls", ".xlsx")):
                            all_data.extend(process_excel(file))
                        else:
                            st.warning(f"Unsupported file format: {file.name}")
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {e}")
                    
                    file_progress.progress((i + 1) / total_files)
                
                file_progress.empty()
                file_status.empty()
            
            if all_data:
                st.info(f"Found {len(all_data)} raw contacts before deduplication")
                
                if dry_run:
                    # Dry run execution
                    dry_run_results, new_count, dup_count, error_count = save_to_firestore(
                        all_data, dry_run=True
                    )
                    
                    # Display dry run results
                    st.subheader("Dry Run Preview")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Would Add New", new_count)
                    col2.metric("Would Skip Duplicates", dup_count)
                    col3.metric("Would Error", error_count)
                    
                    # Show sample records
                    st.write(f"Sample records (first {min(DRY_RUN_MAX_DISPLAY, len(dry_run_results))} shown):")
                    dry_run_df = pd.DataFrame(dry_run_results)
                    st.dataframe(dry_run_df.head(DRY_RUN_MAX_DISPLAY))
                    
                    # Add option to proceed with real upload
                    if st.button("✅ Proceed with Actual Upload"):
                        with st.spinner("Uploading contacts..."):
                            new_count, dup_count, error_count = save_to_firestore(all_data)
                            st.success(
                                f"Upload completed! "
                                f"New: {new_count}, Duplicates: {dup_count}, Errors: {error_count}"
                            )
                else:
                    # Normal execution
                    with st.spinner("Uploading contacts..."):
                        new_count, dup_count, error_count = save_to_firestore(all_data)
                        st.success(
                            f"Upload completed! "
                            f"New: {new_count}, Duplicates: {dup_count}, Errors: {error_count}"
                        )
                
                # Download option
                excel_file, df = generate_sms_excel(all_data)
                st.download_button(
                    "📥 Download Processed Contacts", 
                    data=excel_file, 
                    file_name="processed_contacts.xlsx",
                    help="Download all processed contacts in Excel format"
                )
                
                with st.expander("📊 Processed Data Preview"):
                    st.dataframe(df.head())
            else:
                st.warning("No valid phone numbers found in the uploaded files")

with tabs[1]:
    st.header("Daily SMS List Generator")
    
    if st.button("🔄 Generate Today's SMS List"):
        with st.spinner("Generating SMS list..."):
            sms_data = get_eligible_sms_contacts()
            
            if sms_data:
                sms_excel, sms_df = generate_sms_excel(sms_data)
                
                st.success(f"Prepared {len(sms_data)} contacts for messaging")
                st.download_button(
                    "📥 Download SMS List", 
                    data=sms_excel, 
                    file_name=f"sms_list_{datetime.now().strftime('%Y%m%d')}.xlsx"
                )
                
                with st.expander("👀 View SMS List"):
                    st.dataframe(sms_df)
            else:
                st.warning("No new contacts to message today!")

with tabs[2]:
    st.header("Messaging Dashboard")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.slider(
            "Show data from last N days", 
            min_value=1, 
            max_value=365, 
            value=30
        )
    with col2:
        show_raw_data = st.checkbox("Show raw data")
    
    df = load_dashboard_data(days_back)
    
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        
        # Summary metrics
        st.subheader("📊 Summary Statistics")
        today = pd.to_datetime("today").normalize()
        
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Messages Today", 
            df[df["date"].dt.floor('d') == today].shape[0]
        )
        col2.metric(
            "Messages Last 7 Days", 
            df[df["date"] >= today - pd.Timedelta(days=7)].shape[0]
        )
        col3.metric("Total Messages", df.shape[0])
        
        # Time series chart
        st.subheader("📈 Messages Over Time")
        daily_counts = df.groupby(df["date"].dt.date).size().reset_index(name="Count")
        
        chart = alt.Chart(daily_counts).mark_bar().encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("Count:Q", title="Messages Sent"),
            tooltip=["date:T", "Count"]
        ).properties(
            height=400,
            title=f"Messages Sent (Last {days_back} Days)"
        )
        st.altair_chart(chart, use_container_width=True)
        
        # Top contacts
        st.subheader("🏆 Top Messaged Contacts")
        top_n = st.slider("Show top N contacts", 5, 20, 10)
        top_contacts = df["phone"].value_counts().nlargest(top_n).reset_index()
        top_contacts.columns = ["Phone", "Messages"]
        
        bar_chart = alt.Chart(top_contacts).mark_bar().encode(
            x=alt.X("Messages:Q", title="Message Count"),
            y=alt.Y("Phone:N", title="Phone Number", sort="-x"),
            tooltip=["Phone", "Messages"]
        ).properties(
            height=500,
            title=f"Top {top_n} Most Messaged Contacts"
        )
        st.altair_chart(bar_chart, use_container_width=True)
        
        # Raw data
        if show_raw_data:
            st.subheader("📝 Raw Message Data")
            st.dataframe(df.sort_values("date", ascending=False))
    else:
        st.info("No message data found for the selected period")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        font-size: small;
        color: gray;
        text-align: center;
    }
    </style>
    <div class="footer">
    Contact Management System • v1.1 • Last updated: {date}
    </div>
    """.format(date=datetime.now().strftime("%Y-%m-%d")),
    unsafe_allow_html=True
)