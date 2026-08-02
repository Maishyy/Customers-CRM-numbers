"""Tests for the last-upload-status feature."""

from datetime import datetime, timedelta

from fakes import FakeDB

from firebase_ops import get_last_upload_run
from utils import format_relative_time


class TestGetLastUploadRun:
    def test_returns_most_recent_run(self):
        db = FakeDB()
        db.stores["upload_runs"] = {
            "r1": {
                "timestamp": datetime(2026, 1, 1),
                "file_count": 1,
                "extracted_contacts": 10,
                "added_to_database": 5,
            },
            "r2": {
                "timestamp": datetime(2026, 6, 1),
                "file_count": 2,
                "extracted_contacts": 20,
                "added_to_database": 15,
            },
        }
        run = get_last_upload_run(db)
        assert run["Timestamp"] == datetime(2026, 6, 1)
        assert run["Extracted Contacts"] == 20
        assert run["Added To Database"] == 15

    def test_no_runs_returns_none(self):
        assert get_last_upload_run(FakeDB()) is None

    def test_no_db_returns_none(self):
        assert get_last_upload_run(None) is None


class TestFormatRelativeTime:
    def test_none_is_never(self):
        assert format_relative_time(None) == "Never"

    def test_just_now(self):
        now = datetime(2026, 6, 1, 12, 0, 0)
        assert format_relative_time(now - timedelta(seconds=30), now) == "Just now"

    def test_minutes_ago(self):
        now = datetime(2026, 6, 1, 12, 0, 0)
        assert format_relative_time(now - timedelta(minutes=5), now) == "5 minutes ago"

    def test_singular_minute(self):
        now = datetime(2026, 6, 1, 12, 0, 0)
        assert format_relative_time(now - timedelta(minutes=1), now) == "1 minute ago"

    def test_hours_ago(self):
        now = datetime(2026, 6, 1, 12, 0, 0)
        assert format_relative_time(now - timedelta(hours=3), now) == "3 hours ago"

    def test_yesterday(self):
        now = datetime(2026, 6, 2, 9, 0, 0)
        assert format_relative_time(now - timedelta(days=1), now) == "Yesterday"

    def test_days_ago(self):
        now = datetime(2026, 6, 5, 9, 0, 0)
        assert format_relative_time(now - timedelta(days=3), now) == "3 days ago"

    def test_falls_back_to_date_after_a_week(self):
        now = datetime(2026, 6, 10, 9, 0, 0)
        then = now - timedelta(days=10)
        assert format_relative_time(then, now) == then.strftime("%Y-%m-%d")

    def test_future_timestamp_treated_as_just_now(self):
        now = datetime(2026, 6, 1, 12, 0, 0)
        assert format_relative_time(now + timedelta(seconds=5), now) == "Just now"
