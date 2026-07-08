"""Tests for Firestore write operations using in-memory fakes."""

from fakes import FakeDB, make_phone

from firebase_ops import apply_sms_delivery_report, save_to_firestore


class TestSaveToFirestore:
    def test_saves_new_contacts(self):
        db = FakeDB()
        new, dupes = save_to_firestore(db, [("+254712345678", "John Kamau")])
        assert (new, dupes) == (1, 0)
        saved = db.stores["contacts"]["254712345678"]
        assert saved["client_name"] == "John Kamau"
        assert saved["first_name"] == "John"
        assert saved["last_name"] == "Kamau"
        assert saved["transaction_count"] == 1

    def test_returning_customer_updates_recency_and_count(self):
        db = FakeDB({"254712345678": {
            "phone_number": "+254712345678",
            "client_name": "John",
            "transaction_count": 1,
        }})
        new, dupes = save_to_firestore(db, [("+254712345678", "John")])
        assert (new, dupes) == (0, 1)
        saved = db.stores["contacts"]["254712345678"]
        assert saved["transaction_count"] == 2
        assert "last_transaction_date" in saved
        # Profile data must survive the merge.
        assert saved["client_name"] == "John"

    def test_returning_customer_without_prior_count(self):
        db = FakeDB({"254712345678": {"phone_number": "+254712345678"}})
        save_to_firestore(db, [("+254712345678", "John")])
        assert db.stores["contacts"]["254712345678"]["transaction_count"] == 1

    def test_skips_invalid_contacts(self):
        db = FakeDB()
        new, dupes = save_to_firestore(db, [("+254444444444", "Repeaty")])
        assert (new, dupes) == (0, 0)
        assert db.stores["contacts"] == {}

    def test_empty_input(self):
        assert save_to_firestore(FakeDB(), []) == (0, 0)
        assert save_to_firestore(None, [("+254712345678", "J")]) == (0, 0)

    def test_caller_supplied_existing_numbers_skip_prefetch(self):
        # When the caller passes the existing set, no collection scan runs
        # and duplicates are still detected.
        db = FakeDB({"254712345678": {"phone_number": "+254712345678"}})
        new, dupes = save_to_firestore(
            db,
            [("+254712345678", "John"), ("+254798765432", "Jane Wanjiku")],
            existing_numbers={"+254712345678"},
        )
        assert (new, dupes) == (1, 1)
        assert "254798765432" in db.stores["contacts"]

    def test_batches_are_chunked_at_500(self):
        db = FakeDB()
        data = [(make_phone(i), "Test User") for i in range(600)]
        new, dupes = save_to_firestore(db, data)
        assert (new, dupes) == (600, 0)
        assert len(db.stores["contacts"]) == 600
        assert all(size <= 500 for size in db.commit_sizes)
        assert sum(db.commit_sizes) == 600


class TestApplySmsDeliveryReport:
    @staticmethod
    def row(phone, category, status="DeliveredToTerminal"):
        return {
            "Phone": phone,
            "Raw Status": status,
            "Delivery Status": status,
            "Category": category,
            "Promotional Allowed": category == "promotional",
            "Non Promotional Allowed": category in {"promotional", "non_promotional"},
            "Suppressed": category == "failed",
        }

    def test_promotional_creates_missing_contact(self):
        db = FakeDB()
        stats = apply_sms_delivery_report(
            db, [self.row("+254712345678", "promotional")])
        assert stats["created"] == 1
        assert stats["processed"] == 1
        saved = db.stores["contacts"]["254712345678"]
        assert saved["promotional_allowed"] is True
        assert saved["sms_suppressed"] is False
        assert saved["source"] == "sms_report"

    def test_failed_does_not_create_missing_contact(self):
        db = FakeDB()
        stats = apply_sms_delivery_report(
            db, [self.row("+254712345678", "failed", "AbsentSubscriber")])
        assert stats["skipped"] == 1
        assert stats["created"] == 0
        assert "254712345678" not in db.stores["contacts"]

    def test_failed_suppresses_existing_contact(self):
        db = FakeDB({"254712345678": {
            "phone_number": "+254712345678", "client_name": "John",
        }})
        stats = apply_sms_delivery_report(
            db, [self.row("+254712345678", "failed", "AbsentSubscriber")])
        assert stats["suppressed"] == 1
        assert stats["updated"] == 1
        saved = db.stores["contacts"]["254712345678"]
        assert saved["sms_suppressed"] is True
        assert saved["sms_suppression_reason"] == "AbsentSubscriber"
        # Existing profile data must survive (merge, not overwrite).
        assert saved["client_name"] == "John"

    def test_unrecognized_rows_not_applied(self):
        db = FakeDB()
        stats = apply_sms_delivery_report(
            db, [self.row("+254712345678", "unrecognized", "Weird")])
        assert stats["unrecognized"] == 1
        assert stats["processed"] == 0
        assert "254712345678" not in db.stores["contacts"]

    def test_rows_without_phone_skipped(self):
        stats = apply_sms_delivery_report(
            FakeDB(), [self.row("", "promotional")])
        assert stats["skipped"] == 1

    def test_empty_report(self):
        stats = apply_sms_delivery_report(FakeDB(), [])
        assert stats["processed"] == 0
        assert stats["messages_updated"] == 0

    def test_import_run_is_logged(self):
        db = FakeDB()
        apply_sms_delivery_report(db, [self.row("+254712345678", "promotional")])
        logged = db.collections["sms_report_imports"].added
        assert len(logged) == 1
        assert logged[0]["rows_read"] == 1
        assert "messages_updated" in logged[0]

    def test_existing_numbers_set_avoids_per_row_reads(self):
        db = FakeDB({"254712345678": {
            "phone_number": "+254712345678", "client_name": "John",
        }})
        stats = apply_sms_delivery_report(
            db,
            [
                self.row("+254712345678", "promotional"),
                self.row("+254798765432", "promotional"),
                # Same new phone again with a failed status: must count as
                # an update of the row created above, not a skip.
                self.row("+254798765432", "failed", "AbsentSubscriber"),
            ],
            existing_numbers={"+254712345678"},
        )
        assert stats["updated"] == 2
        assert stats["created"] == 1
        assert stats["skipped"] == 0

    def test_large_report_chunked_at_500(self):
        db = FakeDB()
        rows = [self.row(make_phone(i), "promotional") for i in range(600)]
        stats = apply_sms_delivery_report(db, rows)
        assert stats["created"] == 600
        assert all(size <= 500 for size in db.commit_sizes)
