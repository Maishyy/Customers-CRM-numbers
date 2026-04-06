# main.py
import streamlit as st
import logging
from firebase_setup import setup_firebase
from tabs.upload_tab import render_upload_tab
from tabs.sms_tab import render_sms_tab
from tabs.dashboard_tab import render_dashboard_tab
from tabs.quality_tab import render_quality_tab
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")]
)
logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(layout="wide", page_title="Sequid Hardware Contact System")

def main():
    # Setup Firebase
    db = setup_firebase()
    if db is None:
        st.stop()
    
    # Create tabs
    tabs = st.tabs(["Upload & Sync", "Daily SMS List", "Dashboard", "Data Quality"])
    
    # Render each tab
    with tabs[0]:
        render_upload_tab(db)
    
    with tabs[1]:
        render_sms_tab(db)
    
    with tabs[2]:
        render_dashboard_tab(db)
    
    with tabs[3]:
        render_quality_tab(db)

if __name__ == "__main__":
    main()