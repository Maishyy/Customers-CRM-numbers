import streamlit as st
import pandas as pd
from datetime import datetime

from firebase_admin import firestore

from config import COOLDOWN_DAYS
from firebase_ops import apply_sms_delivery_report, log_message
from processors import generate_standard_excel, parse_sms_delivery_report, process_file_with_duplicate_checks

def is_promotional_sms_allowed(contact_data):
    if not contact_data:
        return True
    if contact_data.get("sms_suppressed"):
        return False
    if contact_data.get("sms_category") == "non_promotional":
        return False
    return contact_data.get("promotional_allowed", True)

def has_successful_sms_report(contact_data):
    return bool(contact_data) and contact_data.get("sms_category") == "promotional"

def render_sms_tab(db):
    st.subheader("Generate Daily SMS List")

    with st.expander("Upload SMS Delivery Report"):
        st.caption("Use provider reports to mark contacts as promotional, non-promotional, or failed/suppressed.")
        report_files = st.file_uploader(
            "Upload SMS report",
            type=["csv", "xls", "xlsx"],
            accept_multiple_files=True,
            key="sms_report_uploader"
        )

        if report_files:
            report_rows = []
            report_issues = []
            for report_file in report_files:
                rows, issues = parse_sms_delivery_report(report_file)
                report_rows.extend(rows)
                report_issues.extend([f"{report_file.name}: {issue}" for issue in issues])

            if report_issues:
                for issue in report_issues:
                    st.warning(issue)

            if report_rows:
                report_df = pd.DataFrame(report_rows)
                st.dataframe(report_df.head(100))

                category_counts = report_df["Category"].value_counts().to_dict()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Promotional", category_counts.get("promotional", 0))
                c2.metric("Non Promotional", category_counts.get("non_promotional", 0))
                c3.metric("Failed/Suppressed", category_counts.get("failed", 0))
                c4.metric("Unrecognized", category_counts.get("unrecognized", 0))

                if st.button("Apply SMS Report to Database"):
                    with st.spinner("Updating contact SMS statuses..."):
                        stats = apply_sms_delivery_report(db, report_rows, report_files)
                    st.success(
                        "SMS report applied: "
                        f"{stats['created']} created, {stats['updated']} updated, "
                        f"{stats['suppressed']} suppressed, {stats['skipped']} skipped."
                    )
                    if stats["unrecognized"]:
                        st.warning(f"{stats['unrecognized']} rows had unrecognized statuses and were not applied.")
            else:
                st.info("No valid phone numbers found in the SMS report.")

    with st.expander("Messaging Settings"):
        new_customers_only = st.checkbox(
            "Only include new customers",
            value=True,
            help="When enabled, only phone numbers not already in the database will be included."
        )
        include_report_approved_existing = st.checkbox(
            "Include existing contacts with successful SMS report status",
            value=True,
            help="Allows DeliveredToTerminal and SentToNetwork contacts back into the promotional list."
        )
        cooldown_days = st.slider(
            "Minimum days between messages",
            min_value=1,
            max_value=30,
            value=COOLDOWN_DAYS,
            help="Customers won't receive messages more frequently than this",
            disabled=new_customers_only
        )
        if new_customers_only:
            st.caption("Cooldown is ignored while sending to new customers only.")

    uploaded_files = st.file_uploader(
        "Upload today's statements",
        type=["pdf", "csv", "xls", "xlsx"],
        accept_multiple_files=True,
        key="daily_sms_uploader"
    )

    if uploaded_files and st.button("Generate Today's SMS List"):
        with st.spinner("Processing statements..."):
            all_contacts = []
            for file in uploaded_files:
                file_contacts = process_file_with_duplicate_checks(file)
                all_contacts.extend(file_contacts)

            unique_contacts = {p: n for p, n in all_contacts if p}
            st.info(f"Processed {len(uploaded_files)} files with {len(unique_contacts)} unique contacts")

            sms_list = []
            batch = db.batch()
            contacts_ref = db.collection("contacts")
            now = datetime.now()

            for phone, name in unique_contacts.items():
                doc_ref = contacts_ref.document(phone.replace("+", ""))
                doc = doc_ref.get()

                eligible = False
                if new_customers_only:
                    eligible = not doc.exists
                    if doc.exists and include_report_approved_existing:
                        contact_data = doc.to_dict()
                        eligible = (
                            has_successful_sms_report(contact_data)
                            and is_promotional_sms_allowed(contact_data)
                        )
                elif doc.exists:
                    contact_data = doc.to_dict()
                    if not is_promotional_sms_allowed(contact_data):
                        eligible = False
                    else:
                        last_trans = contact_data.get("last_transaction_date")
                        if isinstance(last_trans, datetime):
                            delta = now - last_trans
                            eligible = delta.days > cooldown_days
                        else:
                            eligible = True
                else:
                    eligible = True

                if doc.exists and new_customers_only:
                    contact_data = doc.to_dict()
                    if not is_promotional_sms_allowed(contact_data):
                        eligible = False

                if eligible:
                    sms_list.append((phone, name))

                    update_data = {
                        "last_transaction_date": firestore.SERVER_TIMESTAMP,
                        "phone_number": phone,
                        "client_name": name
                    }

                    if not doc.exists:
                        name_parts = name.split() if name else []
                        update_data.update({
                            "first_name": name_parts[0] if name_parts else "",
                            "last_name": " ".join(name_parts[1:]) if len(name_parts) > 1 else "",
                            "source": "upload",
                            "timestamp": firestore.SERVER_TIMESTAMP,
                            "promotional_allowed": True,
                            "non_promotional_allowed": True,
                            "sms_suppressed": False,
                        })

                    batch.set(doc_ref, update_data, merge=True)

            batch.commit()

            for phone, name in sms_list:
                log_message(db, phone, name)

            if sms_list:
                if new_customers_only:
                    st.success(f"{len(sms_list)} new customers eligible for messaging.")
                else:
                    st.success(f"{len(sms_list)} contacts eligible for messaging (cooldown: {cooldown_days} days)")

                sms_excel, sms_df = generate_standard_excel(sms_list)
                st.download_button(
                    "Download SMS List",
                    data=sms_excel,
                    file_name=f"sms_list_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                st.dataframe(sms_df)
            else:
                if new_customers_only:
                    st.info("No new customers found to message from these statements.")
                else:
                    st.info("No eligible contacts to message today from these statements.")
