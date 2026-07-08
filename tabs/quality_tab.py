# tabs/quality_tab.py
import pandas as pd
import streamlit as st

from data_cache import contacts_dataframe
from utils import validate_contact


def render_quality_tab(db):
    st.subheader("Data Quality Analysis")
    st.caption("Runs on a cached copy of the contact list (refreshed every 5 minutes).")

    if st.button("Run Quality Check"):
        with st.spinner("Analyzing database..."):
            df = contacts_dataframe(db)

        if df.empty:
            st.info("No contacts in the database yet.")
            return

        phones = df["Phone"].fillna("")
        names = df["Name"].fillna("")

        total_contacts = len(df)
        invalid_phones = sum(
            1 for phone, name in zip(phones, names)
            if not validate_contact(phone, name or "")
        )
        missing_names = int((names.str.strip() == "").sum())

        phone_counts = phones[phones != ""].value_counts()
        duplicates = phone_counts[phone_counts > 1]

        st.metric("Total Contacts", total_contacts)

        col1, col2 = st.columns(2)
        col1.metric("Invalid Phone Numbers", invalid_phones)
        col2.metric("Contacts Missing Names", missing_names)

        st.metric("Duplicate Phone Numbers", len(duplicates))

        st.subheader("Phone Prefix Distribution")
        prefixes = phones[phones.str.startswith("+254") & (phones.str.len() >= 6)].str[4:6]
        if not prefixes.empty:
            prefix_df = prefixes.value_counts().rename("Count").to_frame()
            st.bar_chart(prefix_df)
        else:
            st.info("No +254 numbers found to chart.")

        if len(duplicates) > 0:
            st.subheader("Sample Duplicate Entries")
            sample = duplicates.head(10)
            for phone, count in sample.items():
                st.write(f"{phone}: {count} occurrences")
                dupe_names = df.loc[df["Phone"] == phone, "Name"].fillna("No name")
                for name in dupe_names:
                    st.write(f"- {name or 'No name'}")
