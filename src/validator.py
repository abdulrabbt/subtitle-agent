"""Response validation and retry logic for translation batches."""

import logging
import time

from openai import (
    APIError,
    APITimeoutError,
    RateLimitError,
    APIConnectionError,
    InternalServerError,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
API_RETRIES = 4
RETRY_DELAYS = [2, 4, 8, 16]


def validate_batch_response(
    response_text: str,
    expected_count: int,
) -> tuple[bool, list[str]]:
    """Validate LLM response: check line count matches expected.

    Fuzzy mode: if the LLM returns MORE lines than expected, take only the
    first N lines (extra lines are usually appended commentary/notes).

    Args:
        response_text: Raw response from DeepSeek.
        expected_count: Number of translated lines expected.

    Returns:
        (is_valid, parsed_lines)
    """
    lines = [line.strip() for line in response_text.strip().split("\n")]
    lines = [line for line in lines if line]

    if len(lines) != expected_count:
        logger.warning(
            "Translation count mismatch: got %d lines, expected %d",
            len(lines),
            expected_count,
        )
        # Fuzzy recovery: if we got more lines than expected, take first N
        if len(lines) > expected_count:
            logger.warning(
                "Fuzzy recovery: taking first %d of %d lines (extra lines discarded)",
                expected_count,
                len(lines),
            )
            logger.debug(
                "Raw response (%d lines):\n---\n%s\n---",
                len(lines),
                response_text.strip(),
            )
            return True, lines[:expected_count]
        # Too few lines — can't recover, will retry
        logger.debug(
            "Raw response (%d lines, too few):\n---\n%s\n---",
            len(lines),
            response_text.strip(),
        )
        return False, lines

    return True, lines


def _retry_on_api_error(fn, *args, **kwargs):
    """Call a function with exponential backoff retry on transient API errors.

    Retries on: timeout, rate limit, connection, server errors.
    Impossible on: bad request (400) — our fault, retrying won't help.

    Returns the function's return value on success.
    Raises RuntimeError if all API retries are exhausted.
    """
    last_exception = None
    for attempt in range(API_RETRIES):
        try:
            return fn(*args, **kwargs)
        except (APITimeoutError, RateLimitError, APIConnectionError, InternalServerError) as e:
            last_exception = e
            if attempt < API_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                logger.warning(
                    "%s on API attempt %d/%d — retrying in %ds...",
                    type(e).__name__,
                    attempt + 1,
                    API_RETRIES,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "%s on final API attempt %d/%d — giving up",
                    type(e).__name__,
                    attempt + 1,
                    API_RETRIES,
                )
        except APIError as e:
            # Generic APIError — retry if it looks transient, raise otherwise
            status = getattr(e, "status_code", None)
            if status and 400 <= status < 500 and status != 429:
                raise
            last_exception = e
            if attempt < API_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                logger.warning(
                    "APIError (status %s) on attempt %d/%d — retrying in %ds...",
                    status,
                    attempt + 1,
                    API_RETRIES,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "APIError on final attempt %d/%d — giving up",
                    attempt + 1,
                    API_RETRIES,
                )

    raise RuntimeError(
        f"API call failed after {API_RETRIES} retries. "
        f"Last error: {last_exception}"
    )


def translate_with_retry(
    translator_fn,
    entries: list[str],
    system_prompt: str,
    batch_prompt_template: str,
) -> list[str]:
    """Call translator with automatic retry on API errors and validation failures.

    Two-layer retry:
      1. API errors (timeout, rate limit, connection, 5xx) → exponential backoff
      2. Validation failures (wrong line count) → up to MAX_RETRIES attempts

    Returns list of translated lines. Raises RuntimeError if all retries fail.
    """
    expected = len(entries)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug(
                "Validation attempt %d/%d: sending %d entries to DeepSeek:\n%s",
                attempt,
                MAX_RETRIES,
                expected,
                "\n".join(entries),
            )
            response = _retry_on_api_error(
                translator_fn, entries, system_prompt, batch_prompt_template
            )
            is_valid, lines = validate_batch_response(response, expected)

            if is_valid:
                # Restore natural line breaks flattened with " / " during serialization
                lines = [line.replace(" / ", "\n") for line in lines]
                return lines

            logger.warning(
                "Validation failed on attempt %d/%d. Retrying...",
                attempt,
                MAX_RETRIES,
            )
            logger.debug(
                "Raw API response (attempt %d):\n---\n%s\n---",
                attempt,
                response,
            )

        except RuntimeError:
            # API retries exhausted inside _retry_on_api_error — re-raise
            raise

    raise RuntimeError(
        f"Translation failed after {MAX_RETRIES} validation retries. "
        f"Checkpoint saved — you can resume from this point."
    )
