# streamlit_app.py
# Thin wrapper so deployments configured with either entry point
# (Streamlit Cloud defaults to streamlit_app.py) run the same app.
from main import main

main()
