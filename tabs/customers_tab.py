# tabs/customers_tab.py
import pandas as pd
import streamlit as st

import re

from data_cache import clear_caches, contacts_dataframe
from firebase_ops import (
    add_contact_note,
    add_manual_contact,
    get_contact,
    get_message_history,
    load_contact_notes,
    update_contact,
)
from processors import generate_standard_excel
from tabs.dashboard_tab import _coerce_datetime


def search_contacts(contacts_df, query, limit=50):
    """Filter the contacts dataframe by phone fragment or name.

    Pure function over the cached dataframe so searching costs no
    database reads. Returns a list of (phone, name) tuples.
    """
    query = (query or "").strip().lower()
    if not query or contacts_df.empty:
        return []

    digits = re.sub(r"\D", "", query)
    phones = contacts_df["Phone"].fillna("")
    names = contacts_df["Name"].fillna("")

    if digits:
        # Stored numbers are +254...; searches typed in local 07xx/01xx
        # format must match them too.
        variants = {digits}
        if digits.startswith("0"):
            variants.add("254" + digits[1:])
        phone_digits = phones.str.replace(r"\D", "", regex=True)
        mask = phone_digits.apply(lambda p: any(v in p for v in variants))
    else:
        mask = names.str.lower().str.contains(re.escape(query), na=False)

    matched = contacts_df[mask].head(limit)
    return [
        (row_phone, row_name or "")
        for row_phone, row_name in zip(matched["Phone"], matched["Name"])
        if row_phone
    ]


def _segment_definitions(contacts_df, now):
    """Return {segment label: dataframe} built from contact recency/frequency."""
    created = contacts_df["Created At"]
    last_txn = contacts_df["Last Transaction"]
    suppressed = contacts_df["Suppressed"].fillna(False).astype(bool)
    not_suppressed = ~suppressed

    return {
        "New this month": contacts_df[
            not_suppressed & (created >= now - pd.Timedelta(days=30))
        ],
        "Active last 14 days": contacts_df[
            not_suppressed & (last_txn >= now - pd.Timedelta(days=14))
        ],
        "Repeat customers (2+ transactions)": contacts_df[
            not_suppressed & (contacts_df["Transactions"].fillna(1) >= 2)
        ],
        "Inactive 60+ days": contacts_df[
            not_suppressed & last_txn.notna() & (last_txn < now - pd.Timedelta(days=60))
        ],
        "Missing names": contacts_df[
            contacts_df["Name"].fillna("").str.strip() == ""
        ],
        "Suppressed / do not message": contacts_df[suppressed],
    }


def _render_segments(db):
    st.caption(
        "Segments are computed from recency and repeat-transaction counts. "
        "Suppressed contacts are excluded from the messaging segments."
    )
    # Persist across reruns so download-button clicks don't clear the view.
    if st.button("Build Segments"):
        st.session_state["segments_built"] = True
    if not st.session_state.get("segments_built"):
        return

    with st.spinner("Loading contacts..."):
        contacts_df = contacts_dataframe(db)

    if contacts_df.empty:
        st.info("No contacts in the database yet.")
        return

    contacts_df["Created At"] = _coerce_datetime(contacts_df["Created At"])
    contacts_df["Last Transaction"] = _coerce_datetime(contacts_df["Last Transaction"])
    now = pd.Timestamp.now()

    for label, seg_df in _segment_definitions(contacts_df, now).items():
        count = seg_df["Phone"].nunique()
        with st.expander(f"{label}: {count} contacts"):
            if seg_df.empty:
                st.info("No contacts in this segment.")
                continue
            st.dataframe(
                seg_df[["Phone", "Name", "Last Transaction", "Transactions"]].head(100),
                use_container_width=True,
            )
            contact_tuples = [
                (row["Phone"], row["Name"] or "")
                for _, row in seg_df.iterrows() if row["Phone"]
            ]
            excel_file, _ = generate_standard_excel(contact_tuples)
            st.download_button(
                f"Download '{label}' as SMS list",
                data=excel_file,
                file_name=f"segment_{label.lower().replace(' ', '_').replace('/', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"segment_dl_{label}",
            )


def _render_profile(db, phone):
    contact = get_contact(db, phone)
    if not contact:
        st.error("Contact not found — it may have been deleted.")
        return

    st.markdown(f"### {contact.get('client_name') or 'Unnamed contact'}")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Phone", contact.get("phone_number", phone))
    p2.metric("Transactions", int(contact.get("transaction_count", 1) or 1))
    p3.metric("Source", contact.get("source", "unknown") or "unknown")
    p4.metric(
        "SMS Status",
        "Suppressed" if contact.get("sms_suppressed") else (contact.get("sms_category") or "No report"),
    )

    if contact.get("sms_suppressed") and contact.get("sms_suppression_reason"):
        st.warning(f"Suppression reason: {contact['sms_suppression_reason']}")

    with st.form(key=f"edit_contact_{phone}"):
        st.caption("Edit contact")
        new_name = st.text_input("Name", value=contact.get("client_name", "") or "")
        c1, c2 = st.columns(2)
        with c1:
            promotional_allowed = st.checkbox(
                "Promotional SMS allowed",
                value=bool(contact.get("promotional_allowed", True)),
            )
        with c2:
            suppressed = st.checkbox(
                "Suppressed (do not message)",
                value=bool(contact.get("sms_suppressed", False)),
            )
        if st.form_submit_button("Save Changes"):
            fields = {
                "client_name": new_name,
                "promotional_allowed": promotional_allowed,
                "sms_suppressed": suppressed,
            }
            if suppressed and not contact.get("sms_suppressed"):
                fields["sms_suppression_reason"] = "manual"
            elif not suppressed:
                fields["sms_suppression_reason"] = ""
            if update_contact(db, phone, fields):
                clear_caches()
                st.success("Contact updated.")
                st.rerun()
            else:
                st.error("Failed to update contact.")

    st.markdown("#### Interaction Notes")
    with st.form(key=f"add_note_{phone}", clear_on_submit=True):
        note_text = st.text_area("New note", placeholder="e.g. Asked about cement pricing, will call back Friday")
        if st.form_submit_button("Add Note"):
            if add_contact_note(db, phone, note_text):
                st.success("Note added.")
                st.rerun()
            else:
                st.error("Note cannot be empty.")

    notes = load_contact_notes(db, phone)
    if notes:
        for note in notes:
            ts = note["Timestamp"]
            when = ts.strftime("%Y-%m-%d %H:%M") if ts else "(pending)"
            st.markdown(f"- **{when}** — {note['Note']}")
    else:
        st.info("No notes yet for this contact.")

    st.markdown("#### Message History")
    history = get_message_history(db, phone)
    if history:
        st.dataframe(
            pd.DataFrame(history)[["Date", "Status", "Delivery Category"]],
            use_container_width=True,
        )
    else:
        st.info("No messages have been sent to this contact.")


def render_customers_tab(db):
    st.subheader("Customer Lookup")

    query = st.text_input(
        "Search by phone or name",
        placeholder="e.g. 0712... or Jane",
        key="customer_search",
    )

    if query and len(query.strip()) >= 3:
        with st.spinner("Searching..."):
            matches = search_contacts(contacts_dataframe(db), query)

        if not matches:
            st.info("No contacts matched your search.")
        else:
            options = {
                f"{phone} — {name or 'Unnamed'}": phone
                for phone, name in matches
            }
            selection = st.selectbox("Select a customer", list(options.keys()))
            if selection:
                _render_profile(db, options[selection])
    elif query:
        st.caption("Type at least 3 characters to search.")

    st.divider()

    with st.expander("Add Contact Manually"):
        with st.form(key="manual_add_contact", clear_on_submit=True):
            raw_phone = st.text_input("Phone number", placeholder="07XXXXXXXX or +2547XXXXXXXX")
            raw_name = st.text_input("Name (optional)")
            if st.form_submit_button("Add Contact"):
                ok, message = add_manual_contact(db, raw_phone, raw_name)
                if ok:
                    clear_caches()
                    st.success(message)
                else:
                    st.error(message)

    st.divider()
    st.subheader("Customer Segments")
    _render_segments(db)
