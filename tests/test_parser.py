"""Unit tests for src/parser.py."""

import tempfile
import datetime
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from src.parser import parse_srt, write_srt, _wrap_rtl


# ---------------------------------------------------------------------------
# _wrap_rtl
# ---------------------------------------------------------------------------
class TestWrapRtl:
    def test_pure_arabic_gets_wrapped(self):
        result = _wrap_rtl("\u0645\u0631\u062d\u0628\u0627")
        assert result == "\u202B\u0645\u0631\u062d\u0628\u0627\u202C"

    def test_mixed_arabic_and_digits_gets_wrapped(self):
        result = _wrap_rtl("\u0631\u062d\u0644\u0629 847")
        assert result == "\u202B\u0631\u062d\u0644\u0629 847\u202C"

    def test_pure_english_not_wrapped(self):
        result = _wrap_rtl("Hello world")
        assert result == "Hello world"

    def test_empty_string_not_wrapped(self):
        result = _wrap_rtl("")
        assert result == ""

    def test_persian_gets_wrapped(self):
        result = _wrap_rtl("\u0633\u0644\u0627\u0645")
        assert result.startswith("\u202B")
        assert result.endswith("\u202C")


# ---------------------------------------------------------------------------
# parse_srt
# ---------------------------------------------------------------------------
SAMPLE_SRT = """\
1
00:00:01,000 --> 00:00:04,000
Hello world

2
00:00:05,000 --> 00:00:08,500
Goodbye
"""


class TestParseSrt:
    def test_valid_srt_returns_entries(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", encoding="utf-8", delete=False
        ) as f:
            f.write(SAMPLE_SRT)
            f.flush()
            temp_path = f.name

        try:
            entries = parse_srt(temp_path)
            assert len(entries) == 2
            assert entries[0]["index"] == 1
            assert entries[0]["content"] == "Hello world"
            assert isinstance(entries[0]["start"], datetime.timedelta)
            assert isinstance(entries[0]["end"], datetime.timedelta)
            assert entries[1]["index"] == 2
            assert entries[1]["content"] == "Goodbye"
        finally:
            Path(temp_path).unlink()

    def test_file_not_found_raises_error(self):
        with pytest.raises(FileNotFoundError, match="Subtitle file not found"):
            parse_srt("nonexistent_file.srt")


# ---------------------------------------------------------------------------
# write_srt
# ---------------------------------------------------------------------------
class TestWriteSrt:
    def test_roundtrip_preserves_content(self):
        """Write entries then re-parse — content should match."""
        entries = [
            {
                "index": 1,
                "start": datetime.timedelta(seconds=1),
                "end": datetime.timedelta(seconds=4),
                "content": "Hello world",
            },
        ]
        with tempfile.NamedTemporaryFile(
            mode="r", suffix=".srt", encoding="utf-8", delete=False
        ) as f:
            temp_path = f.name

        try:
            write_srt(temp_path, entries)
            parsed = parse_srt(temp_path)
            assert len(parsed) == 1
            assert parsed[0]["content"] == "Hello world"
        finally:
            Path(temp_path).unlink()

    def test_writes_arabic_with_rtl_markers(self):
        """Arabic content is written with RLE/PDF markers."""
        entries = [
            {
                "index": 1,
                "start": datetime.timedelta(seconds=1),
                "end": datetime.timedelta(seconds=4),
                "content": "\u0645\u0631\u062d\u0628\u0627",
            },
        ]
        with tempfile.NamedTemporaryFile(
            mode="r", suffix=".srt", encoding="utf-8", delete=False
        ) as f:
            temp_path = f.name

        try:
            write_srt(temp_path, entries)
            raw = Path(temp_path).read_text(encoding="utf-8")
            assert "\u202B" in raw
            assert "\u202C" in raw
        finally:
            Path(temp_path).unlink()

    def test_roundtrip_with_arabic_preserves_markers(self):
        """Arabic content written then re-parsed retains RLE/PDF markers."""
        entries = [
            {
                "index": 1,
                "start": datetime.timedelta(seconds=1),
                "end": datetime.timedelta(seconds=4),
                "content": "\u0645\u0631\u062d\u0628\u0627 \u0628\u0643",
            },
        ]
        with tempfile.NamedTemporaryFile(
            mode="r", suffix=".srt", encoding="utf-8", delete=False
        ) as f:
            temp_path = f.name

        try:
            write_srt(temp_path, entries)
            parsed = parse_srt(temp_path)
            content = parsed[0]["content"]
            assert content.startswith("\u202B")
            assert content.endswith("\u202C")
        finally:
            Path(temp_path).unlink()

    def test_creates_output_directory(self):
        entries = [
            {
                "index": 1,
                "start": datetime.timedelta(seconds=1),
                "end": datetime.timedelta(seconds=4),
                "content": "Test",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "nested" / "dir" / "output.srt"
            write_srt(str(out), entries)
            assert out.exists()