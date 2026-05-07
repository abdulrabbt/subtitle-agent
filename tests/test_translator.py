"""Unit tests for src/translator.py."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.translator import get_client, translate_batch


class TestGetClient:
    @patch.dict(os.environ, {}, clear=True)
    @patch("src.translator._client", None)
    def test_raises_value_error_when_api_key_missing(self):
        """DEEPSEEK_API_KEY not set — should raise ValueError."""
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY not set"):
            get_client()

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=True)
    @patch("src.translator._client", None)
    @patch("src.translator.OpenAI")
    def test_returns_same_client_on_repeated_calls(self, mock_openai):
        """Singleton: second call returns the same instance."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        client1 = get_client()
        client2 = get_client()

        assert client1 is client2
        assert mock_openai.call_count == 1


class TestTranslateBatch:
    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=True)
    @patch("src.translator._client", None)
    def test_sends_correct_api_call_format(self):
        """Verify translate_batch constructs the correct OpenAI API call."""
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "  translated line 1\ntranslated line 2  "
        mock_client.chat.completions.create.return_value = mock_completion

        with patch("src.translator.OpenAI", return_value=mock_client):
            result = translate_batch(
                ["Hello world", "Goodbye"],
                "System prompt here",
                "Translate {count}:\n{entries}",
            )

        # Verify result has whitespace stripped
        assert result == "translated line 1\ntranslated line 2"

        # Verify API call args
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["model"] == "deepseek-v4-flash"
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][0]["content"] == "System prompt here"
        assert call_kwargs["messages"][1]["role"] == "user"
        assert "Hello world" in call_kwargs["messages"][1]["content"]
        assert "Goodbye" in call_kwargs["messages"][1]["content"]

    @patch.dict(os.environ, {
        "DEEPSEEK_API_KEY": "sk-test",
        "DEEPSEEK_MODEL": "deepseek-custom-model",
    }, clear=True)
    @patch("src.translator._client", None)
    def test_uses_model_from_env_when_set(self):
        """Model defaults to DEEPSEEK_MODEL env var."""
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "test"
        mock_client.chat.completions.create.return_value = mock_completion

        with patch("src.translator.OpenAI", return_value=mock_client):
            translate_batch(
                ["a"],
                "sys",
                "{entries}",
            )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "deepseek-custom-model"