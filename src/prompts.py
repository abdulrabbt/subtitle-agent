"""Translation prompts with dynamic source/target language support."""

# RTL scripts: Arabic, Persian, Hebrew, Urdu, Pashto, Sindhi, Uyghur, Divehi, Yiddish, Kurdish
RTL_LANGS = frozenset({"ar", "fa", "he", "ur", "ps", "sd", "ug", "dv", "yi", "ku"})

# Per-language stylistic guidance for the system prompt
LANG_STYLE_NOTES: dict[str, str] = {
    "ar": "Use 'White Dialect' (Ammiya al-Baida). Strictly avoid formal Fusha and archaic grammar. Use natural, contemporary vocabulary understood by all Arabic speakers. It should sound like a real person in a modern drama—emotional, punchy, and concise.",
    "zh": "Use Simplified Chinese (Mandarin) with natural, conversational phrasing.",
    "ja": "Use natural Japanese with appropriate politeness level for the context.",
    "ko": "Use natural Korean with appropriate speech levels.",
    "fr": "Use natural, conversational French. Prefer shorter constructions over verbosity.",
    "de": "Use natural German word order. Keep sentences efficient.",
    "es": "Use neutral, widely understood Spanish. Avoid regional slang.",
    "pt": "Use Brazilian Portuguese by default. Keep tone conversational.",
    "ru": "Use natural Russian. Preserve nuance in emotional scenes.",
    "it": "Use conversational Italian. Prefer shorter, natural sentences.",
    "tr": "Use natural Turkish with proper vowel harmony and agglutination.",
    "hi": "Use Hindi with natural conversational flow. Transliterate English names.",
    "th": "Use natural Thai. Keep sentences concise to fit subtitle timing.",
    "vi": "Use natural Vietnamese with proper tone markers.",
    "bn": "Use natural Bengali. Maintain the informal/formal register of the original. Keep sentences concise for subtitle timing.",
    "ur": "Use natural Urdu with Nastaliq-appropriate phrasing. Maintain poetic/formal register where the original uses elevated language.",
}


def build_system_prompt(source_lang: str, target_lang: str) -> str:
    """Generate a system prompt tailored to a specific language pair.

    Args:
        source_lang: Source language code (e.g., 'en').
        target_lang: Target language code (e.g., 'ar').

    Returns:
        A formatted system prompt string for the LLM.
    """
    source_label = _lang_label(source_lang)
    target_label = _lang_label(target_lang)
    is_target_rtl = target_lang in RTL_LANGS
    style_note = LANG_STYLE_NOTES.get(target_lang, "")

    rtl_instruction = ""
    if is_target_rtl:
        rtl_instruction = (
            f"\n9. {target_label} is a right-to-left (RTL) language. Ensure translations "
            "flow naturally in RTL order. Punctuation should appear at the start of "
            "the line in the RTL visual position (left side).\n"
        )

    style_line = ""
    if style_note:
        style_line = f"\n5. Style: {style_note}\n"

    prompt = (
        f"You are an expert subtitle translator specializing in {source_label}-to-{target_label} "
        "translation for movies and TV shows.\n"
        "\n"
        "Key Rules:\n"
        "1. Translate ONLY the text — NEVER modify or include timestamps, numbers, or indexes.\n"
        "2. Each line of input is a separate subtitle entry. Translate each independently but "
        "maintain natural dialogue flow.\n"
        "3. Match the original meaning and tone precisely: keep humor, sarcasm, urgency, "
        "and emotion intact.\n"
        f"4. Use natural, idiomatic {target_label} — not robotic or literal translations.{style_line}"
        "6. Handle cultural references naturally: translate idioms with {target_label} equivalents "
        "where possible; transliterate proper names.\n"
        "7. Return ONLY the translated lines — one per line, in the same order, with NO extra "
        "text, NO explanations, and NO numbering.\n"
        "8. Preserve all punctuation exactly as in the source.\n"
        f"{rtl_instruction}"
    )

    # Fix the placeholder in rule 6
    prompt = prompt.replace("{target_label}", target_label)

    return prompt


def _lang_label(code: str) -> str:
    """Map a language code to a human-readable label."""
    labels: dict[str, str] = {
        "en": "English",
        "ar": "Arabic",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "pt": "Portuguese",
        "ru": "Russian",
        "it": "Italian",
        "tr": "Turkish",
        "hi": "Hindi",
        "th": "Thai",
        "vi": "Vietnamese",
        "bn": "Bengali",
        "fa": "Persian",
        "he": "Hebrew",
        "ur": "Urdu",
        "ps": "Pashto",
        "sd": "Sindhi",
        "ug": "Uyghur",
        "dv": "Divehi",
        "yi": "Yiddish",
        "ku": "Kurdish",
    }
    return labels.get(code, code.upper())


BATCH_PROMPT = """Translate the following {count} subtitle entries to the target language.

CRITICAL: Return EXACTLY {count} lines. No more, no less.
- One translation per line, in the same order.
- Do NOT add numbers, labels, commentary, or any extra text.
- Do NOT add blank lines.

{entries}"""


