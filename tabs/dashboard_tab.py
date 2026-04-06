# tabs/dashboard_tab.py
import streamlit as st
from datetime import datetime, timedelta
from firebase_ops import load_message_logs

def render_dashboard_tab(db):
    st.subheader("Message History Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End date", datetime.now())

    if st.button("Refresh Dashboard"):
        df_logs = load_message_logs(db, (end_date - start_date).days)
        if df_logs.empty:
            st.info("No message logs found for selected period.")
        else:
            total_messages = len(df_logs)
            unique_contacts = df_logs["Phone"].nunique()
            success_rate = len(df_logs[df_logs["Status"].str.lower() == "delivered"]) / total_messages if total_messages else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Messages", total_messages)
            c2.metric("Unique Contacts", unique_contacts)
            c3.metric("Success Rate", f"{success_rate:.1%}")

            st.subheader("Message Volume")
            daily_counts = df_logs.groupby("Date Messaged").size()
            st.bar_chart(daily_counts)

            st.subheader("Top Recipients")
            top_contacts = df_logs["Phone"].value_counts().nlargest(10)
            st.bar_chart(top_contacts)

            st.subheader("Message Log")
            st.dataframe(df_logs.sort_values("Date Messaged", ascending=False))

            csv = df_logs.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Full Log CSV",
                data=csv,
                file_name="message_log.csv",
                mime="text/csv"
            )