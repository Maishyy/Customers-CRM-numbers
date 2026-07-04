from io import BytesIO

import pandas as pd
import pytest

from processors import (
    classify_sms_delivery_status,
    extract_contacts,
    extract_from_dataframe,
    extract_structured_statement_contacts,
    generate_standard_excel,
    parse_sms_delivery_report,
    should_exclude_line,
)


class NamedBytesIO(BytesIO):
    """File-like object with a .name, mimicking a Streamlit UploadedFile."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


class TestClassifySmsDeliveryStatus:
    @pytest.mark.parametrize("raw,canonical,category", [
        ("DeliveredToTerminal", "DeliveredToTerminal", "promotional"),
        ("delivered to terminal", "DeliveredToTerminal", "promotional"),
        ("SENTTONETWORK", "SentToNetwork", "promotional"),
        ("SenderName Blacklisted", "SenderName Blacklisted", "non_promotional"),
        ("AbsentSubscriber", "AbsentSubscriber", "failed"),
        ("DeliveryImpossible", "DeliveryImpossible", "failed"),
        ("Unknown", "Unknown", "failed"),
    ])
    def test_known_statuses(self, raw, canonical, category):
        assert classify_sms_delivery_status(raw) == (canonical, category)

    def test_unrecognized_status_preserved(self):
        status, category = classify_sms_delivery_status("SomethingWeird")
        assert status == "SomethingWeird"
        assert category == "unrecognized"

    def test_none_and_empty(self):
        assert classify_sms_delivery_status(None) == ("", "unrecognized")
        assert classify_sms_delivery_status("") == ("", "unrecognized")


class TestShouldExcludeLine:
    def test_hard_words_always_excluded(self):
        assert should_exclude_line("CDM DEPOSIT 0712345678", has_phone=True) is True

    def test_soft_words_excluded_without_phone(self):
        assert should_exclude_line("BANK TRANSFER SUMMARY", has_phone=False) is True

    def test_soft_words_kept_with_phone(self):
        assert should_exclude_line("BANK TRANSFER 0712345678", has_phone=True) is False

    def test_non_string(self):
        assert should_exclude_line(None, has_phone=False) is True


class TestExtractContacts:
    def test_extracts_phone_and_adjacent_name(self):
        results = extract_contacts("JOHN KAMAU 0712345678")
        assert results == [("+254712345678", "John Kamau")]

    def test_deduplicates_same_phone(self):
        text = "JOHN KAMAU 0712345678\nJOHN KAMAU 0712345678"
        results = extract_contacts(text)
        assert len(results) == 1

    def test_hard_excluded_line_skipped(self):
        assert extract_contacts("CHEQUE JOHN KAMAU 0712345678") == []

    def test_blacklisted_number_skipped(self):
        assert extract_contacts("PROMO LINE 0722000000") == []

    def test_no_phone_no_results(self):
        assert extract_contacts("just some words here") == []


class TestExtractFromDataframe:
    def test_extracts_from_rows(self):
        df = pd.DataFrame({
            "Details": ["Payment JANE WANJIKU 0798765432", "no phone here"],
        })
        results = extract_from_dataframe(df)
        phones = [p for p, _ in results]
        assert "+254798765432" in phones
        # The name heuristic keeps up to 3 tokens near the phone, so leading
        # words like "Payment" may be included; the person's name must survive.
        name = dict(results)["+254798765432"]
        assert "Jane Wanjiku" in name


class TestExtractStructuredStatementContacts:
    def test_coop_narration_format(self):
        df = pd.DataFrame({
            "Narration": ["~254712345678~TXN12345~ JOHN KAMAU ~more"],
        })
        results = extract_structured_statement_contacts(df)
        assert results == [("+254712345678", "John Kamau")]

    def test_family_remarks_format(self):
        df = pd.DataFrame({
            "Remarks": ["From 254798765432 JANE WANJIKU Alias Code 99"],
        })
        results = extract_structured_statement_contacts(df)
        assert results == [("+254798765432", "Jane Wanjiku")]

    def test_no_match(self):
        df = pd.DataFrame({"Remarks": ["plain text"]})
        assert extract_structured_statement_contacts(df) == []


class TestParseSmsDeliveryReport:
    def _csv(self, body: str) -> NamedBytesIO:
        return NamedBytesIO(body.encode("utf-8"), "report.csv")

    def test_parses_categories_and_flags(self):
        rows, issues = parse_sms_delivery_report(self._csv(
            "Phone Number,Delivery Description\n"
            "0712345678,DeliveredToTerminal\n"
            "0798765432,AbsentSubscriber\n"
            "0110123456,SenderName Blacklisted\n"
        ))
        assert issues == []
        assert len(rows) == 3
        by_phone = {r["Phone"]: r for r in rows}

        promo = by_phone["+254712345678"]
        assert promo["Category"] == "promotional"
        assert promo["Promotional Allowed"] is True
        assert promo["Suppressed"] is False

        failed = by_phone["+254798765432"]
        assert failed["Category"] == "failed"
        assert failed["Promotional Allowed"] is False
        assert failed["Suppressed"] is True

        nonpromo = by_phone["+254110123456"]
        assert nonpromo["Category"] == "non_promotional"
        assert nonpromo["Non Promotional Allowed"] is True
        assert nonpromo["Promotional Allowed"] is False

    def test_deduplicates_phone_status_pairs(self):
        rows, _ = parse_sms_delivery_report(self._csv(
            "Phone Number,Delivery Description\n"
            "0712345678,DeliveredToTerminal\n"
            "0712345678,DeliveredToTerminal\n"
        ))
        assert len(rows) == 1

    def test_invalid_phone_rows_skipped(self):
        rows, _ = parse_sms_delivery_report(self._csv(
            "Phone Number,Delivery Description\n"
            "not-a-phone,DeliveredToTerminal\n"
        ))
        assert rows == []

    def test_missing_columns_reports_issue(self):
        rows, issues = parse_sms_delivery_report(self._csv(
            "ColA,ColB\nfoo,bar\nbaz,qux\n"
        ))
        assert rows == []
        assert issues, "expected an issue for undetectable phone/status columns"


class TestGenerateStandardExcel:
    def test_tuple_input_round_trip(self):
        buffer, df = generate_standard_excel([
            ("+254712345678", "John Kamau"),
            ("+254798765432", ""),
        ])
        assert list(df["Valid"]) == ["Yes", "Yes"]
        assert df.iloc[0]["Firstname(optional)"] == "John"
        assert df.iloc[0]["Lastname(optional)"] == "Kamau"
        assert df.iloc[0]["Phone or Email"] == "254712345678"

        # The buffer must be a readable xlsx with the same rows.
        round_trip = pd.read_excel(buffer, dtype=str)
        assert len(round_trip) == 2

    def test_tuple_input_deduplicates(self):
        _, df = generate_standard_excel([
            ("+254712345678", "John"),
            ("+254712345678", "John"),
        ])
        assert len(df) == 1

    def test_dict_input_preserves_columns(self):
        _, df = generate_standard_excel([
            {"Name": "John Kamau", "Phone": "+254712345678",
             "Existing": "No", "Valid": "Yes"},
        ])
        assert df.iloc[0]["Existing"] == "No"
        assert df.iloc[0]["Firstname(optional)"] == "John"

    def test_invalid_phone_marked_invalid(self):
        _, df = generate_standard_excel([("+254812345678", "John")])
        assert df.iloc[0]["Valid"] == "No"
