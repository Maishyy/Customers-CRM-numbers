# data_cache.py
# Cached Firestore reads shared across tabs. Firestore bills per document
# read, so full-collection scans are cached for a few minutes instead of
# re-run on every tab click.
import streamlit as st

from firebase_ops import (
    get_existing_phone_numbers,
    get_last_upload_run,
    load_contacts_dataframe,
    load_message_logs,
    load_upload_runs,
)

CACHE_TTL_SECONDS = 300
LAST_UPLOAD_CACHE_TTL_SECONDS = 60


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def contacts_dataframe(_db):
    return load_contacts_dataframe(_db)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def existing_phone_numbers(_db):
    return get_existing_phone_numbers(_db)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def message_logs(_db, days):
    return load_message_logs(_db, days)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def upload_runs(_db, days):
    return load_upload_runs(_db, days)


@st.cache_data(ttl=LAST_UPLOAD_CACHE_TTL_SECONDS, show_spinner=False)
def last_upload_run(_db):
    return get_last_upload_run(_db)


def clear_caches():
    """Call after any write so the next read reflects the change."""
    contacts_dataframe.clear()
    existing_phone_numbers.clear()
    message_logs.clear()
    upload_runs.clear()
    last_upload_run.clear()
