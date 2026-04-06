# tabs/quality_tab.py
import streamlit as st
import pandas as pd
from collections import defaultdict

from utils import validate_contact

def render_quality_tab(db):
    st.subheader("Data Quality Analysis")

    if st.button("Run Quality Check"):
        with st.spinner("Analyzing database..."):
            contacts_ref = db.collection("contacts")
            docs = contacts_ref.stream()

            stats = {
                "total_contacts": 0,
                "invalid_phones": 0,
                "missing_names": 0,
                "duplicate_phones": defaultdict(int),
                "phone_prefixes": defaultdict(int)
            }
            phone_counts = defaultdict(int)

            for doc in docs:
                data = doc.to_dict()
                phone = data.get("phone_number", "")
                name = data.get("client_name", "")

                stats["total_contacts"] += 1
                phone_counts[phone] += 1

                if not validate_contact(phone, name or ""):
                    stats["invalid_phones"] += 1
                if not (name or "").strip():
                    stats["missing_names"] += 1

                if phone.startswith("+254") and len(phone) >= 6:
                    pref2 = phone[4:6]
                    stats["phone_prefixes"][pref2] += 1

            stats["duplicate_count"] = sum(1 for count in phone_counts.values() if count > 1)

            st.metric("Total Contacts", stats["total_contacts"])

            col1, col2 = st.columns(2)
            col1.metric("Invalid Phone Numbers", stats["invalid_phones"])
            col2.metric("Contacts Missing Names", stats["missing_names"])

            st.metric("Duplicate Phone Numbers", stats["duplicate_count"])

            st.subheader("Phone Prefix Distribution")
            prefix_df = pd.DataFrame.from_dict(
                stats["phone_prefixes"], orient="index", columns=["Count"]
            ).sort_values("Count", ascending=False)
            st.bar_chart(prefix_df)

            if stats["duplicate_count"] > 0:
                st.subheader("Sample Duplicate Entries")
                duplicates = {p: c for p, c in phone_counts.items() if c > 1}
                sample_dupes = list(duplicates.items())[:10]
                for phone, count in sample_dupes:
                    st.write(f"{phone}: {count} occurrences")
                    dlist = contacts_ref.where("phone_number", "==", phone).stream()
                    for dd in dlist:
                        st.write(f"- {dd.to_dict().get('client_name', 'No name')}")
