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
                "Date Messaged": date_str,
                "Status": d.get("status", "unknown")
            })
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading logs: {str(e)}")
        return pd.DataFrame(columns=["Phone", "Date Messaged", "Status"])

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
