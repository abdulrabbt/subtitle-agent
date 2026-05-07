"""Unit tests for src/validator.py."""

import time
from unittest.mock import MagicMock, patch

import pytest
from openai import (
    APIError,
    APITimeoutError,
    RateLimitError,
    APIConnectionError,
    InternalServerError,
)

from src.validator import (
    _retry_on_api_error,
    validate_batch_response,
    translate_with_retry,
    MAX_RETRIES,
    API_RETRIES,
    RETRY_DELAYS,
)


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


# ---------------------------------------------------------------------------
# _retry_on_api_error
# ---------------------------------------------------------------------------
class TestRetryOnApiError:
    """Tests for the _retry_on_api_error() low-level retry wrapper."""

    def test_succeeds_on_first_attempt(self):
        """No error raised — returns result, called exactly once."""
        func = MagicMock(return_value="success")
        result = _retry_on_api_error(func, "arg1", kw=42)
        assert result == "success"
        func.assert_called_once_with("arg1", kw=42)

    def test_retries_on_api_timeout_error(self):
        """APITimeoutError -> retries with backoff, succeeds on 2nd try."""
        func = MagicMock(side_effect=[APITimeoutError(None), "ok"])
        with patch("time.sleep", return_value=None) as mock_sleep:
            result = _retry_on_api_error(func)
        assert result == "ok"
        assert func.call_count == 2
        mock_sleep.assert_called_once_with(RETRY_DELAYS[0])

    def test_retries_on_rate_limit_error(self):
        """RateLimitError -> retries with backoff."""
        req = MagicMock()  # httpx.Request mock
        resp = MagicMock(request=req, status_code=429)
        func = MagicMock(side_effect=[
            RateLimitError("too many", response=resp, body=None), "ok"
        ])
        with patch("time.sleep", return_value=None):
            result = _retry_on_api_error(func)
        assert result == "ok"
        assert func.call_count == 2

    def test_retries_on_connection_error(self):
        """APIConnectionError -> retries with backoff."""
        req = MagicMock()  # httpx.Request mock
        func = MagicMock(side_effect=[
            APIConnectionError(message="refused", request=req), "ok"
        ])
        with patch("time.sleep", return_value=None):
            result = _retry_on_api_error(func)
        assert result == "ok"
        assert func.call_count == 2

    def test_retries_on_internal_server_error(self):
        """InternalServerError -> retries with backoff."""
        req = MagicMock()  # httpx.Request mock
        resp = MagicMock(request=req, status_code=500)
        func = MagicMock(side_effect=[
            InternalServerError("500 oops", response=resp, body=None), "ok"
        ])
        with patch("time.sleep", return_value=None):
            result = _retry_on_api_error(func)
        assert result == "ok"
        assert func.call_count == 2

    def test_retries_on_5xx_api_error(self):
        """APIError with 502 status -> retries."""
        err = APIError("server error", None, body=None)
        err.status_code = 502
        func = MagicMock(side_effect=[err, "ok"])
        with patch("time.sleep", return_value=None):
            result = _retry_on_api_error(func)
        assert result == "ok"
        assert func.call_count == 2

    def test_retries_on_429_api_error(self):
        """APIError with 429 status (rate limit) -> retries."""
        err = APIError("rate limited", None, body=None)
        err.status_code = 429
        func = MagicMock(side_effect=[err, "ok"])
        with patch("time.sleep", return_value=None):
            result = _retry_on_api_error(func)
        assert result == "ok"
        assert func.call_count == 2

    def test_raises_immediately_on_4xx_not_429(self):
        """APIError with 400/401/403 -> NO retry, raises immediately."""
        err = APIError("bad request", None, body=None)
        err.status_code = 400
        func = MagicMock(side_effect=[err, "should not reach"])
        with patch("time.sleep", return_value=None) as mock_sleep:
            with pytest.raises(APIError, match="bad request"):
                _retry_on_api_error(func)
        assert func.call_count == 1
        mock_sleep.assert_not_called()

    def test_retries_on_api_error_no_status(self):
        """APIError without status_code -> retries (conservative: assume transient)."""
        err = APIError("unknown", None, body=None)
        func = MagicMock(side_effect=[err, "ok"])
        with patch("time.sleep", return_value=None):
            result = _retry_on_api_error(func)
        assert result == "ok"
        assert func.call_count == 2

    def test_raises_runtime_error_after_all_retries(self):
        """All API attempts fail -> RuntimeError with last error message."""
        func = MagicMock(side_effect=[
            APITimeoutError(None),
            APITimeoutError(None),
            APITimeoutError(None),
            APITimeoutError(None),
        ])
        with patch("time.sleep", return_value=None):
            with pytest.raises(RuntimeError, match="API call failed after"):
                _retry_on_api_error(func)
        assert func.call_count == API_RETRIES

    def test_exponential_backoff_delays(self):
        """Verifies delays progress: RETRY_DELAYS[0], [1], [2], [3]."""
        func = MagicMock(side_effect=[
            APITimeoutError(None),
            APITimeoutError(None),
            APITimeoutError(None),
            "finally ok",
        ])
        with patch("time.sleep", return_value=None) as mock_sleep:
            _retry_on_api_error(func)
        assert mock_sleep.call_count == 3
        expected_calls = [
            ((RETRY_DELAYS[0],),),
            ((RETRY_DELAYS[1],),),
            ((RETRY_DELAYS[2],),),
        ]
        assert mock_sleep.call_args_list == expected_calls

    def test_mixed_retryable_then_non_retryable(self):
        """First error retryable, second is 401 -> stops immediately."""
        err = APIError("unauthorized", None, body=None)
        err.status_code = 401
        func = MagicMock(side_effect=[
            APITimeoutError(None),
            err,
            "should not reach",
        ])
        with patch("time.sleep", return_value=None):
            with pytest.raises(APIError, match="unauthorized"):
                _retry_on_api_error(func)
        assert func.call_count == 2  # called twice, not more
