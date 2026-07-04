from io import BytesIO

import pytest

from utils import (
    clean_name,
    format_phone_number,
    safe_file_size,
    validate_contact,
    validate_contact_strict,
)


class TestFormatPhoneNumber:
    @pytest.mark.parametrize("raw,expected", [
        ("0712345678", "+254712345678"),
        ("712345678", "+254712345678"),
        ("254712345678", "+254712345678"),
        ("+254712345678", "+254712345678"),
        ("+254 712 345 678", "+254712345678"),
        ("0110123456", "+254110123456"),
        ("110123456", "+254110123456"),
    ])
    def test_valid_variants(self, raw, expected):
        assert format_phone_number(raw) == expected

    @pytest.mark.parametrize("raw", [
        None,
        "",
        "12345",
        "abc",
        "0812345678",      # 08x is not a valid Kenyan mobile prefix
        "25471234567",     # 11 digits
        "2547123456789",   # 13 digits
        "0012345678",
    ])
    def test_invalid_inputs(self, raw):
        assert format_phone_number(raw) is None

    @pytest.mark.parametrize("raw", [
        "+254722000000",
        "0722000000",
        "254722000000",
        "0766145780",       # blacklisted in raw 07xx form; must be caught post-normalization
        "+254766145780",
    ])
    def test_blacklisted_numbers_rejected(self, raw):
        assert format_phone_number(raw) is None


class TestCleanName:
    def test_strips_noise_tokens(self):
        assert clean_name("MPESA JOHN DOE") == "John Doe"

    def test_strips_transaction_codes(self):
        assert clean_name("TX123ABC456 MARY") == "Mary"

    def test_deduplicates_tokens_case_insensitively(self):
        assert clean_name("john John DOE") == "John Doe"

    def test_caps_at_three_tokens(self):
        assert clean_name("Alpha Beta Gamma Delta") == "Alpha Beta Gamma"

    def test_rejects_single_letter_tokens(self):
        assert clean_name("J") == ""

    def test_non_string_input(self):
        assert clean_name(None) == ""
        assert clean_name(123) == ""

    def test_strips_digits_and_symbols(self):
        assert clean_name("J0hn D@e 42") == ""  # leftover single letters are rejected


class TestValidateContact:
    def test_valid_contact(self):
        assert validate_contact("+254712345678", "John Doe") is True

    def test_valid_without_name(self):
        assert validate_contact("+254712345678", "") is True

    def test_wrong_length(self):
        assert validate_contact("+25471234567", "John") is False

    def test_wrong_prefix(self):
        assert validate_contact("+254812345678", "John") is False

    def test_name_with_short_token(self):
        assert validate_contact("+254712345678", "John D") is False

    def test_strict_rejects_repetitive_digits(self):
        # Only 3 distinct digits across the whole number.
        assert validate_contact_strict("+254444444444", "John") is False

    def test_strict_accepts_normal_numbers(self):
        assert validate_contact_strict("+254712345678", "John") is True


class TestSafeFileSize:
    def test_bytesio_fallback(self):
        buf = BytesIO(b"hello world")
        assert safe_file_size(buf) == 11
