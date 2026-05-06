"""SRT file parser and writer. Timestamps are NEVER modified — only text content is replaced."""

import srt
from pathlib import Path


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
            content=e["content"],
        )
        for e in entries
    ]

    output_text = srt.compose(srt_entries)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8")