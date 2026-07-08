# firebase_ops.py
import re

import pandas as pd
from io import BytesIO
import logging
from datetime import datetime, timedelta
from collections import defaultdict

from firebase_admin import firestore
from utils import clean_name, format_phone_number, validate_contact_strict
from processors import generate_standard_excel

# How a delivery-report category translates to a message log status.
MESSAGE_STATUS_BY_CATEGORY = {
    "promotional": "delivered",
    "non_promotional": "blocked",
    "failed": "failed",
}

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

def save_to_firestore(db, data, existing_numbers=None):
    """Save contacts to Firestore with duplicate prevention.

    Pass existing_numbers (a set of +254... strings) to skip the
    full-collection prefetch when the caller already has it.
    """
    if not db or not data:
        return 0, 0

    coll = db.collection("contacts")
    batch = db.batch()
    new_count = 0
    duplicate_count = 0
    writes = 0

    if existing_numbers is None:
        existing_numbers = set()
        try:
            for doc in coll.select(["phone_number"]).stream():
                existing_numbers.add(doc.get("phone_number"))
        except Exception as e:
            logger.warning(f"Prefetch existing numbers failed: {e}")
    else:
        existing_numbers = set(existing_numbers)

    for phone, name in {(p, n) for p, n in data if p}:
        if not validate_contact_strict(phone, name):
            continue

        doc_ref = coll.document(phone.replace("+", ""))

        if phone in existing_numbers:
            # Returning customer: they appeared in a fresh statement, so
            # refresh recency and count the repeat transaction.
            batch.set(doc_ref, {
                "last_transaction_date": firestore.SERVER_TIMESTAMP,
                "transaction_count": firestore.Increment(1),
            }, merge=True)
            duplicate_count += 1
            writes += 1
            if writes % 500 == 0:
                batch.commit()
                batch = db.batch()
            continue

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
            "last_transaction_date": firestore.SERVER_TIMESTAMP,
            "transaction_count": 1,
        })
        new_count += 1
        writes += 1
        existing_numbers.add(phone)

        if writes % 500 == 0:
            batch.commit()
            batch = db.batch()

    if writes % 500 != 0:
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

def update_message_statuses_from_report(db, report_rows):
    """Resolve queued messages_sent entries using a delivery report.

    Matches queued log entries by phone number and sets their status to
    delivered/blocked/failed so dashboard metrics reflect reality.
    Returns the number of message logs updated.
    """
    if not db or not report_rows:
        return 0

    category_by_phone = {
        row["Phone"]: row["Category"]
        for row in report_rows
        if row.get("Phone") and row.get("Category") in MESSAGE_STATUS_BY_CATEGORY
    }
    if not category_by_phone:
        return 0

    updated = 0
    writes = 0
    batch = db.batch()
    try:
        docs = db.collection("messages_sent").where("status", "==", "queued").stream()
        for doc in docs:
            d = doc.to_dict()
            category = category_by_phone.get(d.get("phone_number"))
            if not category:
                continue
            batch.update(doc.reference, {
                "status": MESSAGE_STATUS_BY_CATEGORY[category],
                "delivery_category": category,
                "status_updated_at": firestore.SERVER_TIMESTAMP,
            })
            updated += 1
            writes += 1
            if writes % 500 == 0:
                batch.commit()
                batch = db.batch()
        if writes % 500 != 0:
            batch.commit()
    except Exception as e:
        logger.error(f"Failed to update message statuses from report: {e}")

    return updated

def apply_sms_delivery_report(db, report_rows, files=None, existing_numbers=None):
    """Apply SMS delivery report categories to contacts without deleting history.

    Pass existing_numbers (a set of +254... strings) to decide
    created-vs-updated from memory instead of one Firestore read per row.
    """
    if not db or not report_rows:
        return {
            "processed": 0,
            "created": 0,
            "updated": 0,
            "suppressed": 0,
            "skipped": 0,
            "unrecognized": 0,
            "messages_updated": 0,
        }

    if existing_numbers is not None:
        # Copy so created contacts can be tracked without mutating the
        # caller's (possibly cached) set.
        existing_numbers = set(existing_numbers)

    contacts_ref = db.collection("contacts")
    batch = db.batch()
    stats = {
        "processed": 0,
        "created": 0,
        "updated": 0,
        "suppressed": 0,
        "skipped": 0,
        "unrecognized": 0,
        "messages_updated": 0,
    }
    writes = 0

    for row in report_rows:
        phone = row.get("Phone")
        category = row.get("Category")
        if not phone:
            stats["skipped"] += 1
            continue
        if category == "unrecognized":
            stats["unrecognized"] += 1
            continue

        doc_ref = contacts_ref.document(phone.replace("+", ""))
        if existing_numbers is not None:
            exists = phone in existing_numbers
        else:
            exists = doc_ref.get().exists

        # Failed report rows should suppress existing contacts, but should not create new contacts.
        if category == "failed" and not exists:
            stats["skipped"] += 1
            continue

        update_data = {
            "phone_number": phone,
            "sms_last_status": row.get("Delivery Status", ""),
            "sms_last_raw_status": row.get("Raw Status", ""),
            "sms_category": category,
            "sms_last_report_date": firestore.SERVER_TIMESTAMP,
            "promotional_allowed": bool(row.get("Promotional Allowed")),
            "non_promotional_allowed": bool(row.get("Non Promotional Allowed")),
            "sms_suppressed": bool(row.get("Suppressed")),
        }

        if category == "failed":
            update_data["sms_suppression_reason"] = row.get("Delivery Status", "")
            stats["suppressed"] += 1
        else:
            update_data["sms_suppression_reason"] = ""

        if not exists:
            update_data.update({
                "client_name": "",
                "first_name": "",
                "last_name": "",
                "source": "sms_report",
                "timestamp": firestore.SERVER_TIMESTAMP,
                "last_transaction_date": firestore.SERVER_TIMESTAMP,
            })
            stats["created"] += 1
            if existing_numbers is not None:
                existing_numbers.add(phone)
        else:
            stats["updated"] += 1

        batch.set(doc_ref, update_data, merge=True)
        writes += 1
        stats["processed"] += 1

        if writes % 500 == 0:
            batch.commit()
            batch = db.batch()

    if writes % 500 != 0:
        batch.commit()

    stats["messages_updated"] = update_message_statuses_from_report(db, report_rows)

    try:
        file_entries = []
        for file in files or []:
            name = getattr(file, "name", "")
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else "unknown"
            file_entries.append({"name": name, "type": ext})

        db.collection("sms_report_imports").add({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "files": file_entries,
            "rows_read": len(report_rows),
            **stats,
        })
    except Exception as e:
        logger.warning(f"Failed to log SMS report import: {e}")

    return stats

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
                "Transactions": d.get("transaction_count", 1),
                "Suppressed": bool(d.get("sms_suppressed", False)),
                "SMS Category": d.get("sms_category", ""),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        logger.error(f"Error loading contacts dataframe: {str(e)}")
        return pd.DataFrame(columns=[
            "Phone", "Name", "Created At", "Last Transaction", "Source",
            "Transactions", "Suppressed", "SMS Category",
        ])

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

def _contact_doc_ref(db, phone):
    return db.collection("contacts").document(phone.replace("+", ""))

def get_contact(db, phone):
    """Return a contact document as a dict, or None if it doesn't exist."""
    try:
        doc = _contact_doc_ref(db, phone).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        logger.error(f"Error fetching contact {phone}: {e}")
        return None

def update_contact(db, phone, fields):
    """Update a whitelisted set of contact fields; derives first/last name."""
    allowed = {
        "client_name", "promotional_allowed", "non_promotional_allowed",
        "sms_suppressed", "sms_suppression_reason",
    }
    update_data = {k: v for k, v in fields.items() if k in allowed}
    if not update_data:
        return False

    if "client_name" in update_data:
        name = clean_name(update_data["client_name"]) or str(update_data["client_name"]).strip()
        parts = name.split(" ", 1) if name else []
        update_data["client_name"] = name
        update_data["first_name"] = parts[0] if parts else ""
        update_data["last_name"] = parts[1] if len(parts) > 1 else ""

    update_data["updated_at"] = firestore.SERVER_TIMESTAMP
    try:
        _contact_doc_ref(db, phone).set(update_data, merge=True)
        return True
    except Exception as e:
        logger.error(f"Error updating contact {phone}: {e}")
        return False

def add_manual_contact(db, raw_phone, raw_name):
    """Manually register a contact. Returns (ok, message)."""
    phone = format_phone_number(raw_phone)
    if not phone:
        return False, "Invalid or blacklisted phone number."

    name = clean_name(raw_name)
    if raw_name and not name:
        return False, "Name could not be cleaned into a valid form."
    if not validate_contact_strict(phone, name):
        return False, "Contact failed validation."

    doc_ref = _contact_doc_ref(db, phone)
    try:
        if doc_ref.get().exists:
            return False, "This phone number already exists in the database."
        parts = name.split(" ", 1) if name else []
        doc_ref.set({
            "phone_number": phone,
            "client_name": name,
            "first_name": parts[0] if parts else "",
            "last_name": parts[1] if len(parts) > 1 else "",
            "source": "manual",
            "timestamp": firestore.SERVER_TIMESTAMP,
            "last_transaction_date": firestore.SERVER_TIMESTAMP,
            "transaction_count": 1,
            "promotional_allowed": True,
            "non_promotional_allowed": True,
            "sms_suppressed": False,
        })
        return True, f"Added {phone}."
    except Exception as e:
        logger.error(f"Error adding manual contact {phone}: {e}")
        return False, "Database error while adding contact."

def add_contact_note(db, phone, text):
    """Attach a free-text interaction note to a contact."""
    text = (text or "").strip()
    if not text:
        return False
    try:
        _contact_doc_ref(db, phone).collection("notes").add({
            "text": text,
            "timestamp": firestore.SERVER_TIMESTAMP,
        })
        return True
    except Exception as e:
        logger.error(f"Error adding note for {phone}: {e}")
        return False

def load_contact_notes(db, phone):
    """Return a contact's notes, newest first."""
    notes = []
    try:
        for doc in _contact_doc_ref(db, phone).collection("notes").stream():
            d = doc.to_dict()
            notes.append({
                "Timestamp": _normalize_timestamp(d.get("timestamp")),
                "Note": d.get("text", ""),
            })
    except Exception as e:
        logger.error(f"Error loading notes for {phone}: {e}")
    notes.sort(key=lambda n: n["Timestamp"] or datetime.min, reverse=True)
    return notes

def get_message_history(db, phone):
    """Return all message log entries for one phone, newest first."""
    history = []
    try:
        docs = db.collection("messages_sent").where("phone_number", "==", phone).stream()
        for doc in docs:
            d = doc.to_dict()
            ts = _normalize_timestamp(d.get("timestamp"))
            history.append({
                "Timestamp": ts,
                "Date": ts.strftime("%Y-%m-%d") if ts else "",
                "Status": d.get("status", "unknown"),
                "Delivery Category": d.get("delivery_category", ""),
            })
    except Exception as e:
        logger.error(f"Error loading message history for {phone}: {e}")
    history.sort(key=lambda h: h["Timestamp"] or datetime.min, reverse=True)
    return history

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
