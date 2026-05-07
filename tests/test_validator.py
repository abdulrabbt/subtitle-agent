"""Unit tests for src/validator.py."""

from unittest.mock import MagicMock

import pytest

from src.validator import validate_batch_response, translate_with_retry, MAX_RETRIES


# ---------------------------------------------------------------------------
# validate_batch_response
# ---------------------------------------------------------------------------
class TestValidateBatchResponse:
    def test_exact_match_returns_valid(self):
        is_valid, lines = validate_batch_response("line one\nline two\nline three", 3)
        assert is_valid is True
        assert lines == ["line one", "line two", "line three"]

    def test_too_many_lines_fuzzy_recovery(self):
        """LLM returned more lines than expected — take first N."""
        is_valid, lines = validate_batch_response(
            "line one\nline two\nline three\nextra line", 3
        )
        assert is_valid is True
        assert lines == ["line one", "line two", "line three"]

    def test_too_few_lines_returns_invalid(self):
        """LLM returned fewer lines than expected — can't recover."""
        is_valid, lines = validate_batch_response("line one\nline two", 4)
        assert is_valid is False
        assert len(lines) == 2

    def test_strips_empty_lines_from_result(self):
        is_valid, lines = validate_batch_response(
            "line one\n\n\nline two\n\n", 2
        )
        assert is_valid is True
        assert lines == ["line one", "line two"]

    def test_strips_trailing_whitespace_per_line(self):
        is_valid, lines = validate_batch_response(
            "  line one  \n  line two  ", 2
        )
        assert is_valid is True
        assert lines == ["line one", "line two"]

    def test_empty_response_returns_invalid(self):
        is_valid, lines = validate_batch_response("", 3)
        assert is_valid is False
        assert lines == []


# ---------------------------------------------------------------------------
# translate_with_retry
# ---------------------------------------------------------------------------
class TestTranslateWithRetry:
    def test_succeeds_first_attempt(self):
        """Mock returns valid response — should succeed on attempt 1."""
        mock_translator = MagicMock(return_value="\u0645\u0631\u062d\u0628\u0627\n\u0643\u064a\u0641 \u062d\u0627\u0644\u0643")
        result = translate_with_retry(
            mock_translator,
            ["Hello", "How are you"],
            "system prompt",
            "batch {count} {entries}",
        )
        assert result == ["\u0645\u0631\u062d\u0628\u0627", "\u0643\u064a\u0641 \u062d\u0627\u0644\u0643"]
        assert mock_translator.call_count == 1

    def test_restores_newlines_from_slash_separator(self):
        """Multiline entries flattened with ' / ' are restored to \\n."""
        mock_translator = MagicMock(return_value="\u0633\u0637\u0631 \u0623\u0648\u0644 / \u0633\u0637\u0631 \u062b\u0627\u0646\u064a")
        result = translate_with_retry(
            mock_translator,
            ["Line one\nLine two"],
            "sys",
            "batch {count} {entries}",
        )
        assert result == ["\u0633\u0637\u0631 \u0623\u0648\u0644\n\u0633\u0637\u0631 \u062b\u0627\u0646\u064a"]

    def test_retries_on_validation_failure_then_succeeds(self):
        """First response has wrong count, second is valid."""
        mock_translator = MagicMock()
        mock_translator.side_effect = [
            "only one line",  # fails validation (expected 2)
            "line one\nline two",  # passes
        ]
        result = translate_with_retry(
            mock_translator,
            ["a", "b"],
            "sys",
            "batch {count} {entries}",
        )
        assert result == ["line one", "line two"]
        assert mock_translator.call_count == 2

    def test_raises_runtime_error_after_all_retries_exhausted(self):
        """Every attempt returns wrong count — should raise RuntimeError."""
        mock_translator = MagicMock(return_value="single line")
        with pytest.raises(RuntimeError, match="Translation failed after"):
            translate_with_retry(
                mock_translator,
                ["a", "b", "c"],
                "sys",
                "batch {count} {entries}",
            )
        assert mock_translator.call_count == MAX_RETRIES

    def test_translator_exception_on_last_attempt_re_raises(self):
        """Translator raises exception on the final retry — should propagate."""
        mock_translator = MagicMock(side_effect=RuntimeError("API down"))
        with pytest.raises(RuntimeError, match="API down"):
            translate_with_retry(
                mock_translator,
                ["a"],
                "sys",
                "batch {count} {entries}",
            )