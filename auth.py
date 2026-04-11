import hmac

import streamlit as st

def _get_configured_password():
    return st.secrets.get("app_password", "")

def is_authenticated():
    return bool(st.session_state.get("authenticated", False))

def logout():
    st.session_state["authenticated"] = False

def require_login():
    """Block access until the correct app password is provided."""
    configured_password = _get_configured_password()
    if not configured_password:
        st.error(
            "This app requires an app password, but none is configured. "
            "Add 'app_password' to your Streamlit secrets before using the app."
        )
        st.stop()

    if is_authenticated():
        with st.sidebar:
            st.caption("Authenticated")
            if st.button("Log out"):
                logout()
                st.rerun()
        return

    st.title("Sign In Required")
    st.write("Enter the app password to continue.")
    password = st.text_input("App password", type="password")

    if st.button("Sign in", type="primary"):
        if hmac.compare_digest(password, configured_password):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Incorrect password.")

    st.stop()
