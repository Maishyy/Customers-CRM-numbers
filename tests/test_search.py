"""Tests for the cache-based customer search (no database reads)."""

import pandas as pd

from fakes import make_phone

from tabs.customers_tab import search_contacts


def contacts_df(rows=None):
    rows = rows if rows is not None else [
        {"Phone": "+254712345678", "Name": "John Kamau"},
        {"Phone": "+254798765432", "Name": "Jane Wanjiku"},
    ]
    return pd.DataFrame(rows)


class TestSearchContacts:
    def test_search_by_local_phone_format(self):
        matches = search_contacts(contacts_df(), "0712345")
        assert matches == [("+254712345678", "John Kamau")]

    def test_search_by_international_fragment(self):
        matches = search_contacts(contacts_df(), "254798")
        assert matches == [("+254798765432", "Jane Wanjiku")]

    def test_search_by_name_case_insensitive(self):
        matches = search_contacts(contacts_df(), "JANE")
        assert matches == [("+254798765432", "Jane Wanjiku")]

    def test_no_match(self):
        assert search_contacts(contacts_df(), "nonexistent") == []

    def test_empty_query_and_empty_frame(self):
        assert search_contacts(contacts_df(), "   ") == []
        assert search_contacts(pd.DataFrame(columns=["Phone", "Name"]), "jane") == []

    def test_limit_respected(self):
        df = contacts_df([
            {"Phone": make_phone(i), "Name": "Bulk User"} for i in range(80)
        ])
        assert len(search_contacts(df, "bulk", limit=50)) == 50

    def test_missing_names_handled(self):
        df = contacts_df([{"Phone": "+254712345678", "Name": None}])
        assert search_contacts(df, "0712345678") == [("+254712345678", "")]
