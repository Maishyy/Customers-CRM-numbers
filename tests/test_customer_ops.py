"""Tests for the message status loop and customer-page operations."""

from datetime import datetime

from fakes import FakeDB, make_phone

from firebase_ops import (
    add_contact_note,
    add_manual_contact,
    apply_sms_delivery_report,
    get_contact,
    get_message_history,
    load_contact_notes,
    update_contact,
    update_message_statuses_from_report,
)


def report_row(phone, category, status="DeliveredToTerminal"):
    return {
        "Phone": phone,
        "Raw Status": status,
        "Delivery Status": status,
        "Category": category,
        "Promotional Allowed": category == "promotional",
        "Non Promotional Allowed": category in {"promotional", "non_promotional"},
        "Suppressed": category == "failed",
    }


class TestUpdateMessageStatusesFromReport:
    def _db_with_messages(self):
        db = FakeDB()
        db.stores["messages_sent"] = {
            "m1": {"phone_number": "+254712345678", "status": "queued"},
            "m2": {"phone_number": "+254798765432", "status": "queued"},
            "m3": {"phone_number": "+254110123456", "status": "queued"},
            "m4": {"phone_number": "+254712345678", "status": "delivered"},
        }
        return db

    def test_resolves_queued_messages_by_category(self):
        db = self._db_with_messages()
        updated = update_message_statuses_from_report(db, [
            report_row("+254712345678", "promotional"),
            report_row("+254798765432", "failed", "AbsentSubscriber"),
            report_row("+254110123456", "non_promotional", "SenderName Blacklisted"),
        ])
        assert updated == 3
        msgs = db.stores["messages_sent"]
        assert msgs["m1"]["status"] == "delivered"
        assert msgs["m2"]["status"] == "failed"
        assert msgs["m3"]["status"] == "blocked"
        assert msgs["m1"]["delivery_category"] == "promotional"

    def test_already_resolved_messages_untouched(self):
        db = self._db_with_messages()
        update_message_statuses_from_report(
            db, [report_row("+254712345678", "promotional")])
        # m4 was already delivered and must not be re-written.
        assert "delivery_category" not in db.stores["messages_sent"]["m4"]

    def test_unrecognized_and_unknown_phones_ignored(self):
        db = self._db_with_messages()
        updated = update_message_statuses_from_report(db, [
            report_row("+254712345678", "unrecognized", "Weird"),
            report_row("+254700000001", "promotional"),
        ])
        assert updated == 0
        assert db.stores["messages_sent"]["m1"]["status"] == "queued"

    def test_empty_inputs(self):
        assert update_message_statuses_from_report(FakeDB(), []) == 0
        assert update_message_statuses_from_report(None, [report_row("+254712345678", "promotional")]) == 0

    def test_large_update_chunked_at_500(self):
        db = FakeDB()
        db.stores["messages_sent"] = {
            f"m{i}": {"phone_number": make_phone(i), "status": "queued"}
            for i in range(600)
        }
        rows = [report_row(make_phone(i), "promotional") for i in range(600)]
        assert update_message_statuses_from_report(db, rows) == 600
        assert all(size <= 500 for size in db.commit_sizes)

    def test_integrated_into_apply_report(self):
        db = self._db_with_messages()
        stats = apply_sms_delivery_report(
            db, [report_row("+254712345678", "promotional")])
        assert stats["messages_updated"] == 1
        assert db.stores["messages_sent"]["m1"]["status"] == "delivered"


class TestGetContact:
    def test_existing(self):
        db = FakeDB({"254712345678": {"phone_number": "+254712345678", "client_name": "John"}})
        contact = get_contact(db, "+254712345678")
        assert contact["client_name"] == "John"

    def test_missing(self):
        assert get_contact(FakeDB(), "+254712345678") is None


class TestUpdateContact:
    def test_updates_name_and_derives_parts(self):
        db = FakeDB({"254712345678": {"phone_number": "+254712345678", "client_name": "Old"}})
        assert update_contact(db, "+254712345678", {"client_name": "Jane Wanjiku"}) is True
        saved = db.stores["contacts"]["254712345678"]
        assert saved["client_name"] == "Jane Wanjiku"
        assert saved["first_name"] == "Jane"
        assert saved["last_name"] == "Wanjiku"

    def test_suppression_fields(self):
        db = FakeDB({"254712345678": {"phone_number": "+254712345678"}})
        update_contact(db, "+254712345678", {
            "sms_suppressed": True, "sms_suppression_reason": "manual",
        })
        saved = db.stores["contacts"]["254712345678"]
        assert saved["sms_suppressed"] is True
        assert saved["sms_suppression_reason"] == "manual"

    def test_disallowed_fields_ignored(self):
        db = FakeDB({"254712345678": {"phone_number": "+254712345678"}})
        assert update_contact(db, "+254712345678", {"phone_number": "+254700000000"}) is False
        assert db.stores["contacts"]["254712345678"]["phone_number"] == "+254712345678"


class TestAddManualContact:
    def test_adds_valid_contact(self):
        db = FakeDB()
        ok, msg = add_manual_contact(db, "0712345678", "John Kamau")
        assert ok, msg
        saved = db.stores["contacts"]["254712345678"]
        assert saved["source"] == "manual"
        assert saved["client_name"] == "John Kamau"
        assert saved["promotional_allowed"] is True

    def test_rejects_invalid_phone(self):
        ok, msg = add_manual_contact(FakeDB(), "12345", "John")
        assert not ok

    def test_rejects_blacklisted_phone(self):
        ok, _ = add_manual_contact(FakeDB(), "0722000000", "John")
        assert not ok

    def test_rejects_existing_contact(self):
        db = FakeDB({"254712345678": {"phone_number": "+254712345678"}})
        ok, msg = add_manual_contact(db, "0712345678", "John")
        assert not ok
        assert "already exists" in msg


class TestNotes:
    def test_add_and_load_notes(self):
        db = FakeDB({"254712345678": {"phone_number": "+254712345678"}})
        assert add_contact_note(db, "+254712345678", "Asked about cement") is True
        notes = load_contact_notes(db, "+254712345678")
        assert len(notes) == 1
        assert notes[0]["Note"] == "Asked about cement"

    def test_empty_note_rejected(self):
        assert add_contact_note(FakeDB(), "+254712345678", "   ") is False

    def test_notes_sorted_newest_first(self):
        db = FakeDB()
        db.stores["contacts/254712345678/notes"] = {
            "n1": {"text": "older", "timestamp": datetime(2026, 1, 1)},
            "n2": {"text": "newer", "timestamp": datetime(2026, 6, 1)},
        }
        notes = load_contact_notes(db, "+254712345678")
        assert [n["Note"] for n in notes] == ["newer", "older"]


class TestGetMessageHistory:
    def test_history_filtered_and_sorted(self):
        db = FakeDB()
        db.stores["messages_sent"] = {
            "m1": {"phone_number": "+254712345678", "status": "delivered",
                   "timestamp": datetime(2026, 1, 1)},
            "m2": {"phone_number": "+254712345678", "status": "queued",
                   "timestamp": datetime(2026, 6, 1)},
            "m3": {"phone_number": "+254798765432", "status": "queued",
                   "timestamp": datetime(2026, 6, 2)},
        }
        history = get_message_history(db, "+254712345678")
        assert len(history) == 2
        assert history[0]["Status"] == "queued"      # newest first
        assert history[1]["Status"] == "delivered"
        assert history[0]["Date"] == "2026-06-01"

    def test_no_history(self):
        assert get_message_history(FakeDB(), "+254712345678") == []
