"""Response validation and retry logic for translation batches."""

import logging

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


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


def translate_with_retry(
    translator_fn,
    entries: list[str],
    system_prompt: str,
    batch_prompt_template: str,
) -> list[str]:
    """Call translator with automatic retry on validation failure.

    If validation fails, logs the raw API response at DEBUG level so users
    can inspect what the LLM returned. Run with --debug to see these logs.

    Returns list of translated lines. Raises RuntimeError if all retries fail.
    """
    expected = len(entries)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug(
                "Attempt %d/%d: sending %d entries to DeepSeek:\n%s",
                attempt,
                MAX_RETRIES,
                expected,
                "\n".join(entries),
            )
            response = translator_fn(entries, system_prompt, batch_prompt_template)
            is_valid, lines = validate_batch_response(response, expected)

            if is_valid:
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

        except Exception as e:
            logger.error(
                "API error on attempt %d/%d: %s", attempt, MAX_RETRIES, str(e)
            )
            if attempt == MAX_RETRIES:
                raise

    raise RuntimeError(
        f"Translation failed after {MAX_RETRIES} retries. "
        f"Checkpoint saved — you can resume from this point."
    )