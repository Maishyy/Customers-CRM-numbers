from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from data_cache import contacts_dataframe, message_logs, upload_runs

def _coerce_datetime(series):
    if series.empty:
        return series
    result = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return result.dt.tz_convert(None)
    except AttributeError:
        return result
    except TypeError:
        return result

def _day_bounds(start_date, end_date):
    start_dt = pd.Timestamp(start_date).tz_localize(None).normalize()
    end_dt = pd.Timestamp(end_date).tz_localize(None).normalize() + pd.Timedelta(days=1)
    return start_dt, end_dt

def render_dashboard_tab(db):
    st.subheader("CRM Insights Dashboard")

    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Start date", datetime.now() - timedelta(days=30), key="dashboard_start")
    with col2:
        end_date = st.date_input("End date", datetime.now(), key="dashboard_end")
    with col3:
        cooldown_days = st.slider(
            "Follow-up cooldown",
            min_value=1,
            max_value=60,
            value=14,
            help="Used for the eligible-for-follow-up insight."
        )

    if st.button("Refresh Dashboard"):
        start_dt, end_dt = _day_bounds(start_date, end_date)
        now = pd.Timestamp.now()
        days = max((end_dt - start_dt).days, 1)

        contacts_df = contacts_dataframe(db)
        message_df = message_logs(db, days + 30)
        upload_df = upload_runs(db, max(days + 30, 90))

        if contacts_df.empty and message_df.empty and upload_df.empty:
            st.info("No dashboard data available yet.")
            return

        if not contacts_df.empty:
            contacts_df["Created At"] = _coerce_datetime(contacts_df["Created At"])
            contacts_df["Last Transaction"] = _coerce_datetime(contacts_df["Last Transaction"])

        if not message_df.empty and "Timestamp" in message_df.columns:
            message_df["Timestamp"] = _coerce_datetime(message_df["Timestamp"])
            message_filtered = message_df[
                (message_df["Timestamp"] >= start_dt) & (message_df["Timestamp"] < end_dt)
            ].copy()
        else:
            message_filtered = pd.DataFrame(columns=["Phone", "Name", "Timestamp", "Date Messaged", "Status"])

        if not upload_df.empty:
            upload_df["Timestamp"] = _coerce_datetime(upload_df["Timestamp"])
            upload_filtered = upload_df[
                (upload_df["Timestamp"] >= start_dt) & (upload_df["Timestamp"] < end_dt)
            ].copy()
        else:
            upload_filtered = pd.DataFrame(columns=["Timestamp"])

        if not contacts_df.empty:
            created_mask = (contacts_df["Created At"] >= start_dt) & (contacts_df["Created At"] < end_dt)
            transaction_mask = (contacts_df["Last Transaction"] >= start_dt) & (contacts_df["Last Transaction"] < end_dt)
            new_customers = contacts_df.loc[created_mask, "Phone"].nunique()
            returning_customers = contacts_df.loc[
                transaction_mask & ((contacts_df["Created At"].isna()) | (contacts_df["Created At"] < start_dt)),
                "Phone"
            ].nunique()
            eligible_follow_up = contacts_df.loc[
                contacts_df["Last Transaction"].notna() &
                (contacts_df["Last Transaction"] <= (now - pd.Timedelta(days=cooldown_days))),
                "Phone"
            ].nunique()
            missing_names = contacts_df.loc[contacts_df["Name"].fillna("").str.strip() == "", "Phone"].nunique()
            total_contacts = contacts_df["Phone"].nunique()
        else:
            new_customers = returning_customers = eligible_follow_up = missing_names = total_contacts = 0

        total_messages = len(message_filtered)
        unique_recipients = message_filtered["Phone"].nunique() if not message_filtered.empty else 0
        repeat_recipients = (
            message_filtered["Phone"].value_counts().gt(1).sum() if not message_filtered.empty else 0
        )
        success_rate = (
            len(message_filtered[message_filtered["Status"].str.lower() == "delivered"]) / total_messages
            if total_messages else 0
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("New Customers Added", new_customers)
        m2.metric("Returning Customers", returning_customers)
        m3.metric("Eligible for Follow-up", eligible_follow_up)
        m4.metric("Contacts Missing Names", missing_names)

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Total Contacts", total_contacts)
        m6.metric("Messages Sent", total_messages)
        m7.metric("Unique Recipients", unique_recipients)
        m8.metric("Repeat Recipients", repeat_recipients)

        st.subheader("New vs Existing Trend")
        if not upload_filtered.empty:
            trend_df = upload_filtered.copy()
            trend_df["Date"] = trend_df["Timestamp"].dt.date.astype(str)
            trend_chart = (
                trend_df.groupby("Date")[["Added To Database", "Existing Contacts"]]
                .sum()
                .sort_index()
            )
            st.line_chart(trend_chart)
        elif not contacts_df.empty:
            fallback = contacts_df.loc[
                contacts_df["Created At"].notna() &
                (contacts_df["Created At"] >= start_dt) &
                (contacts_df["Created At"] < end_dt)
            ].copy()
            if not fallback.empty:
                fallback["Date"] = fallback["Created At"].dt.date.astype(str)
                fallback_chart = fallback.groupby("Date").size().rename("New Customers")
                st.line_chart(fallback_chart)
            else:
                st.info("No upload trend data yet. This will populate after future confirmed uploads.")
        else:
            st.info("No upload trend data yet.")

        st.subheader("Most Active Purchase Days")
        if not contacts_df.empty and contacts_df["Last Transaction"].notna().any():
            activity = contacts_df.loc[transaction_mask].copy()
            if not activity.empty:
                activity["Date"] = activity["Last Transaction"].dt.date.astype(str)
                active_days = activity.groupby("Date").size().sort_index()
                st.bar_chart(active_days)
            else:
                st.info("No recent transaction activity found for the selected period.")
        else:
            st.info("No transaction activity data available yet.")

        st.subheader("Top Repeat Customers")
        if not message_filtered.empty:
            repeat_df = (
                message_filtered.groupby(["Phone", "Name"]).size().reset_index(name="Messages")
                .sort_values("Messages", ascending=False)
                .head(10)
            )
            st.dataframe(repeat_df, use_container_width=True)
        else:
            st.info("No repeat-customer messaging data found for the selected period.")

        st.subheader("Source File / Source Type Mix")
        if not upload_filtered.empty:
            source_counts = (
                upload_filtered.explode("Source Types")["Source Types"]
                .dropna()
                .replace("", pd.NA)
                .dropna()
                .value_counts()
            )
            if not source_counts.empty:
                st.bar_chart(source_counts)
            else:
                st.info("Upload source types have not been captured yet.")
        else:
            st.info("No upload history found yet. This insight will grow as new uploads are confirmed.")

        st.subheader("Recent Upload Summary")
        if not upload_filtered.empty:
            latest_upload = upload_filtered.sort_values("Timestamp", ascending=False).iloc[0]
            u1, u2, u3, u4 = st.columns(4)
            u1.metric("Latest Extracted", int(latest_upload["Extracted Contacts"]))
            u2.metric("Latest Valid", int(latest_upload["Valid Contacts"]))
            u3.metric("Latest Existing", int(latest_upload["Existing Contacts"]))
            u4.metric("Latest Added", int(latest_upload["Added To Database"]))

            latest_files = latest_upload.get("Files", [])
            if latest_files:
                file_df = pd.DataFrame(latest_files)
                st.dataframe(file_df, use_container_width=True)
        else:
            st.info("No upload summary has been logged yet.")

        st.subheader("Messaging Conversion Proxy")
        c1, c2, c3 = st.columns(3)
        c1.metric("Success Rate", f"{success_rate:.1%}")
        c2.metric("Messages per Recipient", f"{(total_messages / unique_recipients):.2f}" if unique_recipients else "0.00")
        c3.metric("Recipients Messaged More Than Once", repeat_recipients)

        if not message_filtered.empty:
            status_counts = message_filtered["Status"].str.lower().value_counts()
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Delivered", int(status_counts.get("delivered", 0)))
            s2.metric("Failed", int(status_counts.get("failed", 0)))
            s3.metric("Blocked", int(status_counts.get("blocked", 0)))
            s4.metric("Still Queued", int(status_counts.get("queued", 0)))
            if int(status_counts.get("queued", 0)):
                st.caption(
                    "Queued messages resolve to delivered/failed/blocked when you apply "
                    "an SMS delivery report in the Daily SMS List tab."
                )

        if not message_filtered.empty:
            message_volume = message_filtered.copy()
            message_volume["Date"] = message_volume["Timestamp"].dt.date.astype(str)
            daily_messages = message_volume.groupby("Date").size().sort_index()
            st.line_chart(daily_messages)

        st.subheader("Message Log")
        if not message_filtered.empty:
            export_df = message_filtered.sort_values("Timestamp", ascending=False).copy()
            export_df["Date Messaged"] = export_df["Timestamp"].dt.strftime("%Y-%m-%d")
            st.dataframe(export_df[["Phone", "Name", "Date Messaged", "Status"]], use_container_width=True)

            csv = export_df[["Phone", "Name", "Date Messaged", "Status"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Full Log CSV",
                data=csv,
                file_name="message_log.csv",
                mime="text/csv"
            )
        else:
            st.info("No message logs found for the selected period.")
