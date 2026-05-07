"""Unit tests for src/prompts.py."""

from src.prompts import SYSTEM_PROMPT, BATCH_PROMPT


class TestSystemPrompt:
    def test_not_empty(self):
        assert len(SYSTEM_PROMPT) > 0

    def test_contains_key_rules(self):
        assert "translator" in SYSTEM_PROMPT.lower()
        assert "arabic" in SYSTEM_PROMPT.lower()
        assert "subtitle" in SYSTEM_PROMPT.lower()


class TestBatchPrompt:
    def test_not_empty(self):
        assert len(BATCH_PROMPT) > 0

    def test_contains_format_placeholders(self):
        assert "{count}" in BATCH_PROMPT
        assert "{entries}" in BATCH_PROMPT