"""DeepSeek V4 Pro translation client using OpenAI-compatible API."""

import os
from openai import OpenAI
from dotenv import load_dotenv

from src.prompts import build_system_prompt, BATCH_PROMPT

load_dotenv()

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Get or create the OpenAI client configured for DeepSeek."""
    global _client
    if _client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not set. Add it to your .env file."
            )
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def translate_batch(
    entries: list[str],
    system_prompt: str,
    batch_prompt_template: str,
    model: str | None = None,
    temperature: float = 0.3,
) -> str:
    """Send a batch of subtitle entries to the LLM for translation.

    Args:
        entries: List of subtitle texts to translate.
        system_prompt: System-level instructions for the LLM.
        batch_prompt_template: Template string with {count} and {entries} placeholders.
        model: Model name. Defaults to DEEPSEEK_MODEL from .env, then "deepseek-v4-flash".
        temperature: LLM temperature (lower = more consistent).

    Returns:
        Raw response text containing translated lines separated by newlines.
    """
    if model is None:
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    client = get_client()
    count = len(entries)
    numbered = "\n".join(e.replace("\n", " / ") for e in entries)

    batch_prompt = batch_prompt_template.format(count=count, entries=numbered)

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        timeout=180,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": batch_prompt},
        ],
    )

    return response.choices[0].message.content.strip()