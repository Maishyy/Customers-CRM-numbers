# config.py
import re

# Constants
MAX_FILE_SIZE_MB = 10
COOLDOWN_DAYS = 14
# Entries must be in normalized +254 form: format_phone_number checks this set
# only after converting the raw input to +254XXXXXXXXX.
BLACKLISTED_NUMBERS = {
    "+254722000000", "+254000000000", "+254766145780",
}

SMS_PROMOTIONAL_STATUSES = {"DeliveredToTerminal", "SentToNetwork"}
SMS_NON_PROMOTIONAL_STATUSES = {"SenderName Blacklisted"}
SMS_FAILED_STATUSES = {"AbsentSubscriber", "DeliveryImpossible", "Unknown"}

# Regex patterns
PHONE_FLEX = re.compile(
    r'(?:\+?254|0)?\s*(?:7\d{2}|1[0-9]\d)\s*[\s-]?\d{3}[\s-]?\d{3}'
)

BANK_PHONE_REGEX = re.compile(
    r'(?:\+?254|0)?\s*(?:7\d{2}|1[0-9]\d)\s*\d{3}\s*\d{3}'
)

NAME_TOKEN = r"[A-Za-z][A-Za-z\'\-]{1,}"
NAME_PATTERN = re.compile(rf'({NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,2}})')

NOISE_TOKEN = re.compile(r'\b(?:MPESA|M-PESA|PAYBILL|TILL|ACC(?:OUNT)?|REF(?:ERENCE)?|BALANCE|CONFIRMATION|RECEIPT|SALES)\b', re.I)
TXN_CODE = re.compile(r'\b[A-Z0-9]{8,12}\b')
MPESA_CODE_CHUNK = re.compile(r'\bT[A-Z0-9]{8,}\b', re.I)

EXCLUDE_WORDS_HARD = {"CDM", "CHEQUE"}
EXCLUDE_WORDS_SOFT = {"BANK", "TRANSFER", "LTD", "LIMITED", "INTERNATIONAL"}
