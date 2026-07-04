# firebase_setup.py
import os
import json
import logging
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore

logger = logging.getLogger(__name__)

def _initialize_firebase_app(cred=None):
    """Initialize Firebase once and reuse the existing app on reruns."""
    if not firebase_admin._apps:
        if cred is not None:
            firebase_admin.initialize_app(cred)
        else:
            firebase_admin.initialize_app()

def setup_firebase():
    """Initialize Firebase with test mode support"""
    TEST_MODE = st.sidebar.checkbox(
        "Enable Test Mode",
        help="Use local Firebase emulator (set FIRESTORE_EMULATOR_HOST)."
    )

    db = None
    if TEST_MODE:
        os.environ["FIRESTORE_EMULATOR_HOST"] = "localhost:8080"
        try:
            if os.path.exists("local_test_creds.json"):
                _initialize_firebase_app(credentials.Certificate("local_test_creds.json"))
            else:
                _initialize_firebase_app()
            db = firestore.client()
            st.sidebar.warning("TEST MODE ACTIVE - Using local emulator")
        except Exception as e:
            logger.error(f"Firebase (emulator) initialization failed: {str(e)}")
            st.error(
                "Failed to initialize the Firebase emulator. "
                "Make sure the Firestore emulator is running on localhost:8080."
            )
            return None
    else:
        # A previous Test Mode session may have pointed this process at the
        # emulator; clear it so production traffic reaches real Firestore.
        os.environ.pop("FIRESTORE_EMULATOR_HOST", None)
        try:
            if "firebase_creds" not in st.secrets:
                raise KeyError("Missing 'firebase_creds' in Streamlit secrets.")
            firebase_json = json.loads(st.secrets["firebase_creds"])
            if firebase_json.get("project_id") in {"...", "", None}:
                raise ValueError("Streamlit secrets still contain placeholder Firebase values.")
            cred = credentials.Certificate(firebase_json)
            _initialize_firebase_app(cred)
            db = firestore.client()
        except Exception as e:
            logger.error(f"Firebase initialization failed: {str(e)}")
            st.error(
                "Failed to initialize Firebase. Check .streamlit/secrets.toml locally "
                "or the Streamlit Cloud app secrets for a valid 'firebase_creds' entry."
            )
            return None
    
    return db
