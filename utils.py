# utils.py
import re
import logging
from datetime import datetime

from config import BLACKLISTED_NUMBERS

logger = logging.getLogger(__name__)

def _valid_ke_prefix(digits12: str) -> bool:
    """Validate Kenyan phone prefix"""
    if len(digits12) != 12 or not digits12.startswith('254'):
        return False
    after = digits12[3:]
    if after[0] == '7' and after[1].isdigit():
        return True
    if after[0] == '1' and after[1].isdigit():
        return True
    return False

def format_phone_number(raw: str):
    """Standardize phone to +254 format with rigorous validation."""
    if not raw:
        return None
    digits = re.sub(r'\D', '', str(raw))

    # Normalize to 254xxxxxxxxx (12 digits with country code)
    if digits.startswith('0') and len(digits) == 10:
        digits = '254' + digits[1:]
    elif digits.startswith('7') and len(digits) == 9:
        digits = '254' + digits
    elif digits.startswith('1') and len(digits) == 9:
        digits = '254' + digits
    elif digits.startswith('254') and len(digits) == 12:
        pass
    else:
        return None

    if not _valid_ke_prefix(digits):
        return None

    phone = '+' + digits
    if phone in BLACKLISTED_NUMBERS:
        return None
    return phone

def clean_name(name: str) -> str:
    """Clean and normalize human-like names."""
    if not name or not isinstance(name, str):
        return ""

    # Remove obvious noise and refs
    from config import NOISE_TOKEN, TXN_CODE
    name = NOISE_TOKEN.sub(' ', name)
    name = TXN_CODE.sub(' ', name)
    
    # Remove non-letters except space, apostrophe, hyphen
    name = re.sub(r"[^A-Za-z'\-\s]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()

    tokens = [t.capitalize() for t in name.split() if t]
    seen = set()
    uniq = []
    for t in tokens:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            uniq.append(t)

    uniq = uniq[:3]
    cleaned = ' '.join(uniq).strip()

    if not cleaned:
        return ""
    if any(len(tok) < 2 for tok in cleaned.split()):
        return ""
    return cleaned

def validate_contact(phone, name):
    if not phone or len(phone) != 13 or not phone.startswith("+254"):
        return False
    digits = phone[1:]
    if not _valid_ke_prefix(digits):
        return False
    if name:
        parts = name.split()
        if any(len(p) < 2 for p in parts):
            return False
    return True

def validate_contact_strict(phone, name):
    if not validate_contact(phone, name):
        return False
    digits = re.sub(r'\D', '', phone)
    if len(set(digits)) < 4:
        return False
    return True

def format_relative_time(then: datetime, now: datetime = None) -> str:
    """Human-friendly "time ago" string, e.g. '5 minutes ago', 'Yesterday'."""
    if then is None:
        return "Never"

    now = now or datetime.now()
    if then.tzinfo is not None and now.tzinfo is None:
        then = then.replace(tzinfo=None)

    delta = now - then
    seconds = delta.total_seconds()

    if seconds < 0:
        return "Just now"
    if seconds < 60:
        return "Just now"
    if seconds < 3600:
        minutes = int(seconds // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    if seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"

    days = delta.days
    if days == 1:
        return "Yesterday"
    if days < 7:
        return f"{days} days ago"
    return then.strftime("%Y-%m-%d")

def safe_file_size(uploaded_file) -> int:
    """Return file size in bytes in a Streamlit-version-safe way."""
    try:
        return uploaded_file.size
    except Exception:
        try:
            return len(uploaded_file.getbuffer())
        except Exception:
            try:
                uploaded_file.seek(0, 2)
                size = uploaded_file.tell()
                uploaded_file.seek(0)
                return size
            except Exception:
                return 0