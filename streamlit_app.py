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

# -----------------------------
# Streamlit page config first
# -----------------------------
st.set_page_config(layout="wide", page_title="Sequid Hardware Contact System")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')]
)
logger = logging.getLogger(__name__)

# --- Firebase Setup ---
TEST_MODE = st.sidebar.checkbox("Enable Test Mode", help="Use local Firebase emulator")

db = None
if TEST_MODE:
    os.environ["FIRESTORE_EMULATOR_HOST"] = "localhost:8080"
    try:
        if not firebase_admin._apps:
            # In emulator mode, credentials can be omitted; keep local override if user provided it.
            if os.path.exists("local_test_creds.json"):
                firebase_admin.initialize_app(credentials.Certificate("local_test_creds.json"))
            else:
                firebase_admin.initialize_app()
        db = firestore.client()
        st.sidebar.warning("TEST MODE ACTIVE - Using local emulator")
    except Exception as e:
        logger.error(f"Firebase (emulator) initialization failed: {str(e)}")
        st.error("Failed to initialize Firebase emulator. Check logs for details.")
        st.stop()
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
# Match diverse phone formats; we will normalize/validate later.
# Accepts +254 7xx xxx xxx, 07xx xxx xxx, 7xx xxx xxx, 01xx xxx xxx, 11xx (as 011x after normalization)
PHONE_FLEX = re.compile(
    r'(?:\\+?254|0)?\\s*(?:7\\d{2}|1[01]\\d)\\s*[\\s-]?\\d{3}[\\s-]?\\d{3}'
)

# Name patterns: allow 1-3 tokens, letters & apostrophes/hyphens, mixed case.
NAME_TOKEN = r"[A-Za-z][A-Za-z\\'\\-]{1,}"
NAME_PATTERN = re.compile(rf'({NAME_TOKEN}(?:\\s+{NAME_TOKEN}){{0,2}})')

EXCLUDE_WORDS_HARD = {"CDM", "CHEQUE"}
EXCLUDE_WORDS_SOFT = {"BANK", "TRANSFER", "LTD", "LIMITED", "INTERNATIONAL"}
BLACKLISTED_NUMBERS = {"+254722000000", "+254000000000"}
MAX_FILE_SIZE_MB = 10
VALID_KENYAN_PREFIXES = {'70','71','72','73','74','75','76','77','78','79','11'}

# Common MPESA/bank noise tokens to strip when near names
NOISE_TOKEN = re.compile(r'\\b(?:MPESA|M-PESA|PAYBILL|TILL|ACC(?:OUNT)?|REF(?:ERENCE)?|BALANCE|CONFIRMATION|RECEIPT|SALES)\\b', re.I)
TXN_CODE = re.compile(r'\\b[A-Z0-9]{8,12}\\b')  # MPESA/Bank reference-like strings

def safe_file_size(uploaded_file) -> int:
    """Return file size in bytes in a Streamlit-version-safe way."""
    try:
        return uploaded_file.size  # may exist on newer versions
    except Exception:
        try:
            return len(uploaded_file.getbuffer())
        except Exception:
            uploaded_file.seek(0, 2)
            size = uploaded_file.tell()
            uploaded_file.seek(0)
            return size

# --- Helper Functions ---
def format_phone_number(raw: str):
    """Standardize phone to +254 format with rigorous validation."""
    if not raw:
        return None

    digits = re.sub(r'\\D', '', str(raw))
    # Normalize to 2547/25411 form first
    if digits.startswith('0') and len(digits) == 10:
        # 07xxxxxxxx or 01xxxxxxxx -> 2547xxxxxxx / 2541xxxxxxx
        digits = '254' + digits[1:]
    elif digits.startswith('7') and len(digits) == 9:
        digits = '254' + digits
    elif digits.startswith('1') and len(digits) == 9:  # 1xx for 011x lines
        digits = '254' + digits
    elif digits.startswith('254') and len(digits) == 12:
        pass
    else:
        return None

    # Validate Kenyan mobile prefix
    prefix = digits[3:5]  # e.g., 2547|25411 -> '7x' or '11'
    if len(digits) != 12 or prefix not in VALID_KENYAN_PREFIXES:
        return None

    phone = '+' + digits
    if phone in BLACKLISTED_NUMBERS:
        return None
    return phone

def clean_name(name: str) -> str:
    """Clean and normalize someone-like names."""
    if not name or not isinstance(name, str):
        return ""

    # Remove obvious noise
    name = NOISE_TOKEN.sub(' ', name)
    name = TXN_CODE.sub(' ', name)
    # Remove non-letters except space, apostrophe, hyphen
    name = re.sub(r"[^A-Za-z'\\-\\s]", ' ', name)
    # Collapse spaces
    name = re.sub(r'\\s+', ' ', name).strip()

    # Capitalize tokens
    parts = [p for p in name.split(' ') if p]
    parts = [p.capitalize() for p in parts]

    # Remove duplicate tokens keeping order
    seen = set()
    uniq = []
    for p in parts:
        pl = p.lower()
        if pl not in seen:
            seen.add(pl)
            uniq.append(p)

    # Keep max 3 tokens
    uniq = uniq[:3]
    cleaned = ' '.join(uniq).strip()

    # Filter out too short / clearly invalid
    if not cleaned:
        return ""
    if any(len(tok) < 2 for tok in cleaned.split()):
        return ""
    return cleaned

def should_exclude_line(line: str, has_phone: bool) -> bool:
    """Exclude headers or noise lines. Be softer if a phone exists in the line."""
    if not isinstance(line, str):
        return True
    U = line.upper()
    # Hard excludes (regardless)
    if any(w in U for w in EXCLUDE_WORDS_HARD):
        return True
    # Soft excludes only if no phone in the line
    if not has_phone and any(w in U for w in EXCLUDE_WORDS_SOFT):
        return True
    return False

def try_extract_name_near(text_line: str, start_idx: int, end_idx: int) -> str:
    """Grab a plausible name near the phone in the same line (left or right window)."""
    window_left = text_line[max(0, start_idx-50):start_idx]
    window_right = text_line[end_idx:end_idx+50]

    # Prefer left side tokens like "... JOHN DOE 07xx ..."
    m_left = NAME_PATTERN.findall(window_left)
    cand = m_left[-1] if m_left else ''
    if not cand:
        m_right = NAME_PATTERN.findall(window_right)
        cand = m_right[0] if m_right else ''

    cand = clean_name(cand)
    return cand

@lru_cache(maxsize=64)
def extract_contacts(text: str):
    """Extract contacts line-by-line with improved heuristics and validation."""
    results = []
    seen = set()

    # Split into lines to reduce cross-talk
    lines = re.split(r'\\r?\\n+', text)
    prev_name_hint = ""

    for idx, line in enumerate(lines):
        if not line or not isinstance(line, str):
            continue

        # Find phones in line
        phones = list(PHONE_FLEX.finditer(line))
        has_phone = bool(phones)
        if should_exclude_line(line, has_phone):
            # Keep a weak name hint if line looks like a name-only line
            if not has_phone:
                nm = NAME_PATTERN.search(line)
                prev_name_hint = clean_name(nm.group(0)) if nm else prev_name_hint
            continue

        # If many numbers in line (likely ref codes), skip
        digit_groups = re.findall(r'\\d+', line)
        if len(digit_groups) > 4 and not has_phone:
            continue

        for m in phones:
            raw = m.group(0)
            phone = format_phone_number(raw)
            if not phone or phone in seen:
                continue

            # same-line name first
            name = try_extract_name_near(line, m.start(), m.end())

            # fallback: previous line hint if no name yet
            if not name and idx > 0:
                nm_prev = NAME_PATTERN.search(lines[idx-1]) if idx > 0 else None
                if nm_prev:
                    name = clean_name(nm_prev.group(0))

            # last fallback: keep empty name (will still pass if phone valid)
            if name and not validate_contact(phone, name):
                # name may be too short—keep phone-only
                name = ""

            if validate_contact(phone, name or ""):
                results.append((phone, name or ""))
                seen.add(phone)

    return results

def validate_contact(phone, name):
    """Comprehensive contact validation"""
    if not phone or len(phone) != 13 or not phone.startswith("+254"):
        return False

    # Additional phone validation
    prefix = phone[4:6]  # e.g., +2547..., +25411...
    if prefix not in VALID_KENYAN_PREFIXES:
        return False

    # Name validation (optional)
    if name:
        parts = name.split()
        if any(len(p) < 2 for p in parts):
            return False
    return True

def validate_contact_strict(phone, name):
    """Strict validation with duplicate/entropy checks"""
    if not validate_contact(phone, name):
        return False

    # Check for suspicious duplicate patterns
    digits = re.sub(r'\\D', '', phone)
    if len(set(digits)) < 4:  # Too many repeating digits
        return False
    return True

# --- File Processors ---
def process_pdf(file):
    """Process PDF with duplicate prevention and progress tracking"""
    try:
        with pdfplumber.open(file) as pdf:
            total_pages = len(pdf.pages)
            progress_bar = st.progress(0)
            contacts = []
            seen_numbers = set()

            pages = pdf.pages[::2] if total_pages > 20 else pdf.pages

            for i, page in enumerate(pages):
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                if txt:
                    for phone, name in extract_contacts(txt):
                        if phone and phone not in seen_numbers:
                            contacts.append((phone, name))
                            seen_numbers.add(phone)

                progress_bar.progress((i + 1) / len(pages))

            return contacts
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        st.error(f"Error processing PDF: {str(e)}")
        return []

def extract_from_dataframe(df: pd.DataFrame):
    """Extract contacts from dataframe rows"""
    records = []
    for _, row in df.iterrows():
        vals = [str(x) for x in row.values.tolist() if pd.notna(x)]
        line = " ".join(vals)
        # Determine if this "line" has a phone for exclusion logic
        has_phone = bool(PHONE_FLEX.search(line))
        if not should_exclude_line(line, has_phone):
            records.extend(extract_contacts(line))
    return records

def process_csv(file):
    """Process CSV with chunking for large files"""
    try:
        size = safe_file_size(file)
        if size > 5 * 1024 * 1024:  # 5MB
            records = []
            for chunk in pd.read_csv(file, chunksize=10000):
                records.extend(extract_from_dataframe(chunk))
            return records
        return extract_from_dataframe(pd.read_csv(file))
    except Exception as e:
        logger.error(f"CSV processing error: {str(e)}")
        st.error(f"Error processing CSV: {str(e)}")
        return []

def process_excel(file):
    """Process Excel files with multiple sheets"""
    try:
        name = file.name.lower()
        engine = 'openpyxl' if name.endswith('xlsx') else None
        dfs = pd.read_excel(file, sheet_name=None, engine=engine)
        sheets = list(dfs.items())[:3] if len(dfs) > 3 else dfs.items()
        return [contact for _, df in sheets for contact in extract_from_dataframe(df)]
    except Exception as e:
        logger.error(f"Excel processing error: {str(e)}")
        st.error(f"Error processing Excel: {str(e)}")
        return []

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

def process_file_with_duplicate_checks(file):
    """Process file with enhanced duplicate detection"""
    try:
        contacts = process_file(file)

        # First-level deduplication
        unique_contacts = {}
        for phone, name in contacts:
            if not phone:
                continue
            norm_phone = phone.replace("+", "").strip()
            norm_name = clean_name(name)

            if norm_phone in unique_contacts:
                existing_name = unique_contacts[norm_phone][1]
                if len(norm_name) > len(existing_name):
                    unique_contacts[norm_phone] = (phone, norm_name)
            else:
                unique_contacts[norm_phone] = (phone, norm_name)

        return list(unique_contacts.values())
    except Exception as e:
        logger.error(f"Error in duplicate check: {str(e)}")
        return []

# --- Firebase Operations ---
def save_to_firestore(data):
    """Save contacts to Firestore with duplicate prevention"""
    if not db or not data:
        return 0, 0

    collection = db.collection("contacts")
    batch = db.batch()
    new_count = 0
    duplicate_count = 0

    # Preload existing numbers (phones are stored as '+254...')
    existing_numbers = set()
    try:
        docs = collection.select(["phone_number"]).stream()
        for doc in docs:
            existing_numbers.add(doc.get("phone_number"))
    except Exception as e:
        logger.warning(f"Prefetch existing numbers failed: {e}")

    # Process contacts
    for phone, name in {(p, n) for p, n in data if p}:
        if not validate_contact_strict(phone, name):
            continue

        doc_ref = collection.document(phone.replace("+", ""))

        if phone in existing_numbers:
            duplicate_count += 1
            continue

        first = last = ""
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

        # Commit in batches of 500
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
        docs = (db.collection("messages_sent")
                  .where("timestamp", ">=", cutoff)
                  .stream())

        data = []
        for doc in docs:
            d = doc.to_dict()
            phone = d.get("phone_number", "")
            date = d.get("timestamp")
            # Firestore timestamps may be None or have .strftime
            if hasattr(date, 'strftime'):
                date_str = date.strftime("%Y-%m-%d")
            else:
                try:
                    # Firestore Timestamp to datetime
                    date_str = date.to_datetime().strftime("%Y-%m-%d") if date else ""
                except Exception:
                    date_str = ""
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

def get_last_message_dates(phone_numbers):
    """Safe method to get last message dates for a batch of numbers"""
    last_messages = {}
    if not phone_numbers:
        return last_messages

    batch_size = 30  # Firestore 'in' limit is 30
    phone_batches = [phone_numbers[i:i + batch_size] for i in range(0, len(phone_numbers), batch_size)]

    for batch in phone_batches:
        if not batch:
            continue
        try:
            # Query one-by-one for reliability with order_by+limit
            for p in batch:
                docs = (db.collection("messages_sent")
                          .where("phone_number", "==", p)
                          .order_by("timestamp", direction=firestore.Query.DESCENDING)
                          .limit(1)
                          .stream())
                for doc in docs:
                    data = doc.to_dict()
                    last_messages[data["phone_number"]] = data.get("timestamp")
        except Exception as e:
            logger.error(f"Error fetching messages for batch: {str(e)}")
            continue

    return last_messages

# --- Export Functions ---
def generate_standard_excel(data):
    """Generate standardized Excel output with quality checks using openpyxl styles."""
    formatted_data = []
    seen = set()

    for phone, name in data:
        if phone in seen:
            continue
        seen.add(phone)

        first = last = ""
        if name:
            parts = name.strip().split()
            first = parts[0] if parts else ""
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
        df.to_excel(writer, index=False, sheet_name='Contacts')
        # openpyxl styling
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        wb = writer.book
        ws = writer.sheets['Contacts']

        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill("solid", fgColor="4472C4")
        header_alignment = Alignment(vertical="top", wrap_text=True)
        thin = Side(style='thin')
        thin_border = Border(left=thin, right=thin, top=thin, bottom=thin)

        # Apply to header row
        for col_idx, col_name in enumerate(df.columns, start=1):
            cell = ws.cell(row=1, column=col_idx)
            cell.value = col_name
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
            cell.border = thin_border

        # Autosize columns (basic)
        for col_idx, col_name in enumerate(df.columns, start=1):
            max_len = max([len(str(col_name))] + [len(str(v)) for v in df[col_name].astype(str).tolist()])
            ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = min(max_len + 2, 40)

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

# --- UI Tabs ---
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
        oversized = [f.name for f in uploaded_files if safe_file_size(f) > MAX_FILE_SIZE_MB * 1024 * 1024]
        if oversized:
            st.error(f"Oversized files: {', '.join(oversized)}")
        else:
            with st.spinner("Processing files..."):
                all_data = process_files_parallel(uploaded_files) if len(uploaded_files) > 1 else process_file(uploaded_files[0])

            if all_data:
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

                    st.success(f"Added {new_count} new contacts; skipped {duplicate_count} duplicates.")

                    excel_file, df = generate_standard_excel(all_data)
                    st.download_button(
                        "Download Processed Contacts",
                        data=excel_file,
                        file_name="processed_contacts.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    st.dataframe(df.head())
                else:
                    st.error("Failed to generate contact list")

    with col2:
        if st.button("Refresh Contact Count"):
            # Use count aggregation if available; else fallback
            try:
                agg = db.collection("contacts").count().get()
                # Firestore returns list of snapshots or a snapshot depending on SDK; normalize
                if isinstance(agg, list):
                    count = agg[0][0].value
                else:
                    count = getattr(agg, "value", None) or getattr(getattr(agg, "aggregation_results", [{}])[0], "value", None)
                    if isinstance(count, dict) and "integerValue" in count:
                        count = int(count["integerValue"])
                if count is None:
                    raise ValueError("Count unavailable in this SDK; falling back.")
            except Exception:
                count = sum(1 for _ in db.collection("contacts").stream())
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
        with st.spinner("Processing statement..."):
            current_data = process_file_with_duplicate_checks(uploaded_file)
            current_numbers = {p for p, _ in current_data if p}

            if not current_data:
                st.warning("No valid contacts found in this statement")
            else:
                last_messages = get_last_message_dates(list(current_numbers))

                sms_data = []
                now = datetime.now()
                for phone, name in current_data:
                    if not phone:
                        continue
                    last_message = last_messages.get(phone)
                    eligible = True
                    try:
                        if last_message:
                            # Handle Firestore Timestamp or datetime
                            if hasattr(last_message, 'to_datetime'):
                                last_dt = last_message.to_datetime()
                            else:
                                last_dt = last_message if isinstance(last_message, datetime) else None
                            if last_dt:
                                eligible = (now - last_dt).days > 7
                    except Exception:
                        eligible = True

                    if eligible:
                        sms_data.append((phone, name))

                # Deduplicate and log
                final_sms_data = []
                seen_phones = set()
                for phone, name in sms_data:
                    if phone not in seen_phones:
                        final_sms_data.append((phone, name))
                        seen_phones.add(phone)
                        log_message(phone, name)

                if final_sms_data:
                    sms_excel, sms_df = generate_standard_excel(final_sms_data)
                    st.success(f"{len(final_sms_data)} active contacts ready for messaging.")
                    st.download_button(
                        "Download SMS List",
                        data=sms_excel,
                        file_name=f"sms_list_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    st.dataframe(sms_df)
                else:
                    st.info("No eligible contacts to message today from this statement.")

# --- Tab 3: Dashboard ---
with tabs[2]:
    st.subheader("Message History Dashboard")

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
            total_messages = len(df_logs)
            unique_contacts = df_logs["Phone"].nunique()
            success_rate = len(df_logs[df_logs["Status"].str.lower() == "delivered"]) / total_messages if total_messages else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Messages", total_messages)
            c2.metric("Unique Contacts", unique_contacts)
            c3.metric("Success Rate", f"{success_rate:.1%}")

            st.subheader("Message Volume")
            daily_counts = df_logs.groupby("Date Messaged").size()
            st.bar_chart(daily_counts)

            st.subheader("Top Recipients")
            top_contacts = df_logs["Phone"].value_counts().nlargest(10)
            st.bar_chart(top_contacts)

            st.subheader("Message Log")
            st.dataframe(df_logs.sort_values("Date Messaged", ascending=False))

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
            contacts_ref = db.collection("contacts")
            docs = contacts_ref.stream()

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

                if not validate_contact(phone, name or ""):
                    stats["invalid_phones"] += 1

                if not (name or "").strip():
                    stats["missing_names"] += 1

                if phone.startswith("+254") and len(phone) >= 6:
                    prefix = phone[4:6]
                    stats["phone_prefixes"][prefix] += 1

            stats["duplicate_count"] = sum(1 for count in phone_counts.values() if count > 1)

            st.metric("Total Contacts", stats["total_contacts"])

            col1, col2 = st.columns(2)
            col1.metric("Invalid Phone Numbers", stats["invalid_phones"])
            col2.metric("Contacts Missing Names", stats["missing_names"])

            st.metric("Duplicate Phone Numbers", stats["duplicate_count"])

            st.subheader("Phone Prefix Distribution")
            prefix_df = pd.DataFrame.from_dict(
                stats["phone_prefixes"],
                orient="index",
                columns=["Count"]
            ).sort_values("Count", ascending=False)
            st.bar_chart(prefix_df)

            if stats["duplicate_count"] > 0:
                st.subheader("Sample Duplicate Entries")
                duplicates = {p: c for p, c in phone_counts.items() if c > 1}
                sample_dupes = list(duplicates.items())[:10]

                for phone, count in sample_dupes:
                    st.write(f"{phone}: {count} occurrences")
                    docs = contacts_ref.where("phone_number", "==", phone).stream()
                    for doc in docs:
                        st.write(f"- {doc.to_dict().get('client_name', 'No name')}")