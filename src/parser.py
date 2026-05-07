"""SRT file parser and writer. Timestamps are NEVER modified — only text content is replaced."""

import re
import srt
from pathlib import Path

# Unicode range covering Arabic, Persian, Urdu scripts
_ARABIC_RANGE = re.compile(
    r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
)


def _wrap_rtl(text: str) -> str:
    """Wrap text with RLE/PDF markers if it contains Arabic-script characters.

    RLE (\\u202B) forces the renderer to treat the block as right-to-left.
    PDF (\\u202C) pops the directional formatting.

    Without these markers, neutral punctuation (., !, ?, etc.) can flip
    to the wrong side and LTR runs (numbers, English words) can render
    in reverse order when displayed in subtitle players / editors.
    """
    if _ARABIC_RANGE.search(text):
        return f"\u202B{text}\u202C"
    return text


def parse_srt(filepath: str) -> list[dict]:
    """Read an .srt file and return a list of entry dicts.

    Each entry: {index, start_timedelta, end_timedelta, content}
    Timestamps are stored as datetime.timedelta objects and NEVER modified.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {filepath}")

    text = path.read_text(encoding="utf-8")
    entries = list(srt.parse(text))

    return [
        {
            "index": e.index,
            "start": e.start,   # timedelta object
            "end": e.end,       # timedelta object
            "content": e.content,
        }
        for e in entries
    ]


def write_srt(output_path: str, entries: list[dict]) -> None:
    """Write translated entries back to .srt format.

    entries: list of {index, start, end, content} where:
    - start, end are datetime.timedelta objects (pass-through from parse_srt)
    - content is the Arabic translated text
    """
    srt_entries = [
        srt.Subtitle(
            index=e["index"],
            start=e["start"],
            end=e["end"],
            content=_wrap_rtl(e["content"]),
        )
        for e in entries
    ]

    output_text = srt.compose(srt_entries)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8")