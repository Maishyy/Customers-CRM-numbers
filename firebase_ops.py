# firebase_ops.py
import pandas as pd
from io import BytesIO
import logging
from datetime import datetime, timedelta
from collections import defaultdict

from firebase_admin import firestore
from utils import validate_contact_strict
from processors import generate_standard_excel

logger = logging.getLogger(__name__)

def _normalize_timestamp(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_datetime"):
        try:
            return value.to_datetime()
        except Exception:
            return None
    return None

def get_existing_phone_numbers(db):
    """Return all phone numbers currently stored in Firestore."""
    existing_numbers = set()
    if not db:
        return existing_numbers

    try:
        for doc in db.collection("contacts").select(["phone_number"]).stream():
            phone = doc.get("phone_number")
            if phone:
                existing_numbers.add(phone)
    except Exception as e:
        logger.warning(f"Could not load existing phone numbers: {e}")

    return existing_numbers

def save_to_firestore(db, data):
    """Save contacts to Firestore with duplicate prevention."""
    if not db or not data:
        return 0, 0

    coll = db.collection("contacts")
    batch = db.batch()
    new_count = 0
    duplicate_count = 0

    existing_numbers = set()
    try:
        for doc in coll.select(["phone_number"]).stream():
            existing_numbers.add(doc.get("phone_number"))
    except Exception as e:
        logger.warning(f"Prefetch existing numbers failed: {e}")

    for phone, name in {(p, n) for p, n in data if p}:
        if not validate_contact_strict(phone, name):
            continue

        if phone in existing_numbers:
            duplicate_count += 1
            continue

        doc_ref = coll.document(phone.replace("+", ""))

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

        if new_count % 500 == 0:
            batch.commit()
            batch = db.batch()

    if new_count % 500 != 0:
        batch.commit()

    return new_count, duplicate_count

def log_message(db, phone, name):
    try:
        db.collection("messages_sent").add({
            "phone_number": phone,
            "client_name": name,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "status": "queued"
        })
    except Exception as e:
        logger.error(f"Failed to log message for {phone}: {str(e)}")

def log_upload_run(db, files, report_rows, new_count, duplicate_count):
    """Store upload summary data for dashboard insights."""
    if not db:
        return

    try:
        valid_count = sum(1 for row in report_rows if row.get("Valid") == "Yes")
        unique_contacts = len({row.get("Phone") for row in report_rows if row.get("Phone")})
        existing_count = sum(1 for row in report_rows if row.get("Existing") == "Yes")
        new_to_db_count = sum(1 for row in report_rows if row.get("Existing") == "No")
        file_entries = []
        source_types = []

        for file in files or []:
            name = getattr(file, "name", "")
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else "unknown"
            file_entries.append({"name": name, "type": ext})
            source_types.append(ext)

        db.collection("upload_runs").add({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "file_count": len(file_entries),
            "files": file_entries,
            "source_types": source_types,
            "extracted_contacts": len(report_rows),
            "unique_contacts": unique_contacts,
            "valid_contacts": valid_count,
            "existing_contacts": existing_count,
            "new_contacts_in_upload": new_to_db_count,
            "added_to_database": new_count,
            "duplicates_skipped": duplicate_count,
        })
    except Exception as e:
        logger.warning(f"Failed to log upload run: {e}")

def load_message_logs(db, days=30):
    try:
        cutoff = datetime.now() - timedelta(days=days)
        docs = (db.collection("messages_sent")
                .where("timestamp", ">=", cutoff)
                .stream())
        data = []
        for doc in docs:
            d = doc.to_dict()
            phone = d.get("phone_number", "")
            ts = d.get("timestamp")
            if hasattr(ts, 'strftime'):
                date_str = ts.strftime("%Y-%m-%d")
            else:
                try:
                    date_str = ts.to_datetime().strftime("%Y-%m-%d") if ts else ""
                except Exception:
                    date_str = ""
            data.append({
                "Phone": phone,
                "Name": d.get("client_name", ""),
                "Timestamp": _normalize_timestamp(ts),
                "Date Messaged": date_str,
                "Status": d.get("status", "unknown")
            })
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading logs: {str(e)}")
        return pd.DataFrame(columns=["Phone", "Date Messaged", "Status"])

def load_contacts_dataframe(db):
    try:
        rows = []
        for doc in db.collection("contacts").stream():
            d = doc.to_dict()
            rows.append({
                "Phone": d.get("phone_number", ""),
                "Name": d.get("client_name", ""),
                "Created At": _normalize_timestamp(d.get("timestamp")),
                "Last Transaction": _normalize_timestamp(d.get("last_transaction_date")),
                "Source": d.get("source", ""),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        logger.error(f"Error loading contacts dataframe: {str(e)}")
        return pd.DataFrame(columns=["Phone", "Name", "Created At", "Last Transaction", "Source"])

def load_upload_runs(db, days=90):
    try:
        cutoff = datetime.now() - timedelta(days=days)
        rows = []
        docs = db.collection("upload_runs").where("timestamp", ">=", cutoff).stream()
        for doc in docs:
            d = doc.to_dict()
            rows.append({
                "Timestamp": _normalize_timestamp(d.get("timestamp")),
                "File Count": d.get("file_count", 0),
                "Source Types": d.get("source_types", []),
                "Extracted Contacts": d.get("extracted_contacts", 0),
                "Unique Contacts": d.get("unique_contacts", 0),
                "Valid Contacts": d.get("valid_contacts", 0),
                "Existing Contacts": d.get("existing_contacts", 0),
                "New Contacts In Upload": d.get("new_contacts_in_upload", 0),
                "Added To Database": d.get("added_to_database", 0),
                "Duplicates Skipped": d.get("duplicates_skipped", 0),
                "Files": d.get("files", []),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        logger.error(f"Error loading upload runs: {str(e)}")
        return pd.DataFrame(columns=[
            "Timestamp", "File Count", "Source Types", "Extracted Contacts",
            "Unique Contacts", "Valid Contacts", "Existing Contacts",
            "New Contacts In Upload", "Added To Database", "Duplicates Skipped", "Files"
        ])

def get_last_message_dates(db, phone_numbers):
    last_messages = {}
    if not phone_numbers:
        return last_messages

    batch_size = 30
    batches = [phone_numbers[i:i + batch_size] for i in range(0, len(phone_numbers), batch_size)]
    for batch in batches:
        if not batch:
            continue
        try:
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

def download_full_contact_list(db):
    try:
        contacts_ref = db.collection("contacts")
        docs = contacts_ref.stream()
        contact_list = []
        for doc in docs:
            data = doc.to_dict()
            contact_list.append((data.get("phone_number", ""), data.get("client_name", "")))
        return generate_standard_excel(contact_list)
    except Exception as e:
        logger.error(f"Error downloading full contact list: {str(e)}")
        return None, None
