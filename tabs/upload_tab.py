import pandas as pd
import streamlit as st

from config import MAX_FILE_SIZE_MB
from firebase_ops import download_full_contact_list, get_existing_phone_numbers, log_upload_run, save_to_firestore
from processors import generate_standard_excel, process_file, process_files_parallel
from utils import safe_file_size, validate_contact

def build_upload_report_rows(all_data, existing_numbers):
    rows = []
    for phone, name in all_data:
        is_valid = validate_contact(phone, name)
        rows.append({
            "Name": name,
            "Phone": phone,
            "Existing": "Yes" if phone in existing_numbers else "No",
            "Valid": "Yes" if is_valid else "No",
        })
    return rows

def render_upload_tab(db):
    st.subheader("Upload MPESA or Bank Statements")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "csv", "xls", "xlsx"],
        accept_multiple_files=True,
        help=f"Max file size: {MAX_FILE_SIZE_MB}MB each"
    )

    if uploaded_files:
        oversized = [f.name for f in uploaded_files if safe_file_size(f) > MAX_FILE_SIZE_MB * 1024 * 1024]
        if oversized:
            st.error(f"Oversized files: {', '.join(oversized)}")
        else:
            with st.spinner("Processing files..."):
                all_data = process_files_parallel(uploaded_files) if len(uploaded_files) > 1 else process_file(uploaded_files[0])

            if all_data:
                existing_numbers = get_existing_phone_numbers(db)
                report_rows = build_upload_report_rows(all_data, existing_numbers)

                preview_df = pd.DataFrame(report_rows)

                st.subheader("Data Quality Preview")
                st.dataframe(preview_df.head(50))

                unique_count = len({p for p, _ in all_data if p})
                existing_count = sum(1 for row in report_rows if row["Existing"] == "Yes")
                new_count_preview = sum(1 for row in report_rows if row["Existing"] == "No")
                st.success(f"Found {unique_count} unique contacts.")
                st.caption(f"Existing in database: {existing_count} | New to database: {new_count_preview}")

                report_excel, _ = generate_standard_excel(report_rows)
                st.download_button(
                    "Download Extraction Report",
                    data=report_excel,
                    file_name="extraction_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                if st.button("Confirm Upload to Database"):
                    with st.spinner("Saving to database..."):
                        new_count, duplicate_count = save_to_firestore(db, all_data)
                        log_upload_run(db, uploaded_files, report_rows, new_count, duplicate_count)
                    st.success(f"Added {new_count} new contacts; skipped {duplicate_count} duplicates.")

                    excel_file, df = generate_standard_excel(report_rows)
                    st.download_button(
                        "Download Processed Contacts",
                        data=excel_file,
                        file_name="processed_contacts.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.warning("No valid phone numbers found.")

    st.subheader("Contact Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Download Full Contact List"):
            with st.spinner("Preparing full contact list..."):
                excel_file, df = download_full_contact_list(db)
                if excel_file:
                    st.download_button(
                        label="Download Complete Contacts",
                        data=excel_file,
                        file_name="full_contact_list.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    st.dataframe(df.head())
                else:
                    st.error("Failed to generate contact list")

    with col2:
        if st.button("Refresh Contact Count"):
            try:
                agg = db.collection("contacts").count().get()
                if isinstance(agg, list):
                    count = agg[0][0].value
                else:
                    count = getattr(agg, "value", None) or getattr(getattr(agg, "aggregation_results", [{}])[0], "value", None)
                    if isinstance(count, dict) and "integerValue" in count:
                        count = int(count["integerValue"])
                if count is None:
                    raise ValueError("Count unavailable; falling back.")
            except Exception:
                count = sum(1 for _ in db.collection("contacts").stream())
            st.metric("Total Contacts in Database", count)
