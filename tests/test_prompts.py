"""Unit tests for src/prompts.py."""

from src.prompts import (
    build_system_prompt,
    _lang_label,
    BATCH_PROMPT,
    RTL_LANGS,
    LANG_STYLE_NOTES,
)


class TestBuildSystemPrompt:
    def test_not_empty(self):
        prompt = build_system_prompt("en", "ar")
        assert len(prompt) > 0

    def test_contains_translator_and_language_info(self):
        prompt = build_system_prompt("en", "ar")
        assert "translator" in prompt.lower()
        assert "Arabic" in prompt
        assert "English" in prompt
        assert "subtitle" in prompt.lower()

    def test_includes_rtl_instruction_for_arabic(self):
        prompt = build_system_prompt("en", "ar")
        assert "right-to-left" in prompt.lower()

    def test_no_rtl_instruction_for_ltr_language(self):
        prompt = build_system_prompt("en", "fr")
        assert "right-to-left" not in prompt.lower()

    def test_includes_style_note_when_available(self):
        prompt = build_system_prompt("en", "ar")
        assert "White Dialect" in prompt

    def test_no_style_note_when_unavailable(self):
        prompt = build_system_prompt("en", "sv")  # Swedish not in LANG_STYLE_NOTES
        assert "5. Style:" not in prompt

    def test_persian_includes_rtl_instruction(self):
        prompt = build_system_prompt("en", "fa")
        assert "right-to-left" in prompt.lower()

    def test_hebrew_includes_rtl_instruction(self):
        prompt = build_system_prompt("en", "he")
        assert "right-to-left" in prompt.lower()

    def test_source_and_target_labels_in_prompt(self):
        prompt = build_system_prompt("ja", "ko")
        assert "Japanese" in prompt
        assert "Korean" in prompt

    def test_rule_6_placeholder_is_replaced(self):
        """Rule 6 has a {target_label} placeholder that must be resolved."""
        prompt = build_system_prompt("en", "de")
        assert "{target_label}" not in prompt
        assert "German" in prompt


class TestLangLabel:
    def test_known_code_returns_label(self):
        assert _lang_label("en") == "English"
        assert _lang_label("ar") == "Arabic"
        assert _lang_label("zh") == "Chinese"
        assert _lang_label("ja") == "Japanese"
        assert _lang_label("ko") == "Korean"
        assert _lang_label("fr") == "French"
        assert _lang_label("de") == "German"
        assert _lang_label("es") == "Spanish"
        assert _lang_label("pt") == "Portuguese"
        assert _lang_label("ru") == "Russian"
        assert _lang_label("it") == "Italian"
        assert _lang_label("tr") == "Turkish"
        assert _lang_label("hi") == "Hindi"
        assert _lang_label("th") == "Thai"
        assert _lang_label("vi") == "Vietnamese"
        assert _lang_label("bn") == "Bengali"
        assert _lang_label("fa") == "Persian"
        assert _lang_label("he") == "Hebrew"
        assert _lang_label("ur") == "Urdu"
        assert _lang_label("ps") == "Pashto"
        assert _lang_label("sd") == "Sindhi"
        assert _lang_label("ug") == "Uyghur"
        assert _lang_label("dv") == "Divehi"
        assert _lang_label("yi") == "Yiddish"
        assert _lang_label("ku") == "Kurdish"

    def test_unknown_code_returns_uppercased_code(self):
        assert _lang_label("sv") == "SV"
        assert _lang_label("xx") == "XX"
        assert _lang_label("abc") == "ABC"


class TestRtlLangs:
    def test_contains_expected_rtl_languages(self):
        assert "ar" in RTL_LANGS
        assert "fa" in RTL_LANGS
        assert "he" in RTL_LANGS
        assert "ur" in RTL_LANGS
        assert "ps" in RTL_LANGS
        assert "sd" in RTL_LANGS
        assert "ug" in RTL_LANGS
        assert "dv" in RTL_LANGS
        assert "yi" in RTL_LANGS
        assert "ku" in RTL_LANGS

    def test_ltr_languages_not_in_rtl_set(self):
        assert "en" not in RTL_LANGS
        assert "fr" not in RTL_LANGS
        assert "de" not in RTL_LANGS
        assert "zh" not in RTL_LANGS
        assert "ja" not in RTL_LANGS

    def test_is_frozenset(self):
        assert isinstance(RTL_LANGS, frozenset)


class TestLangStyleNotes:
    def test_contains_known_languages(self):
        assert "ar" in LANG_STYLE_NOTES
        assert "zh" in LANG_STYLE_NOTES
        assert "ja" in LANG_STYLE_NOTES
        assert "ko" in LANG_STYLE_NOTES
        assert "fr" in LANG_STYLE_NOTES
        assert "de" in LANG_STYLE_NOTES
        assert "es" in LANG_STYLE_NOTES
        assert "pt" in LANG_STYLE_NOTES
        assert "ru" in LANG_STYLE_NOTES
        assert "it" in LANG_STYLE_NOTES
        assert "tr" in LANG_STYLE_NOTES
        assert "hi" in LANG_STYLE_NOTES
        assert "th" in LANG_STYLE_NOTES
        assert "vi" in LANG_STYLE_NOTES
        assert "bn" in LANG_STYLE_NOTES
        assert "ur" in LANG_STYLE_NOTES

    def test_unknown_language_not_in_notes(self):
        assert "sv" not in LANG_STYLE_NOTES
        assert "xx" not in LANG_STYLE_NOTES


class TestBatchPrompt:
    def test_not_empty(self):
        assert len(BATCH_PROMPT) > 0

    def test_contains_format_placeholders(self):
        assert "{count}" in BATCH_PROMPT
        assert "{entries}" in BATCH_PROMPT