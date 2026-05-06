"""JSON checkpoint manager for resume-from-failure support."""

import json
from pathlib import Path
from datetime import datetime, timezone


def get_checkpoint_path(input_path: str) -> str:
    """Generate checkpoint path from input file path.
    Example: TheMatrix.srt -> TheMatrix.srt.checkpoint.json
    """
    return f"{input_path}.checkpoint.json"


def load_checkpoint(input_path: str) -> dict | None:
    """Load a checkpoint if it exists. Returns None if no checkpoint found."""
    checkpoint_path = get_checkpoint_path(input_path)
    path = Path(checkpoint_path)
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    # Verify checkpoint matches this input file
    if data.get("input_path") != input_path:
        return None
    return data


def save_checkpoint(
    input_path: str,
    current_index: int,
    total_entries: int,
    translated: list[str],
) -> None:
    """Save progress to checkpoint JSON file."""
    checkpoint_path = get_checkpoint_path(input_path)
    checkpoint = {
        "input_path": input_path,
        "current_index": current_index,
        "total_entries": total_entries,
        "translated": translated,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    Path(checkpoint_path).write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def delete_checkpoint(input_path: str) -> None:
    """Remove checkpoint file after successful completion."""
    checkpoint_path = get_checkpoint_path(input_path)
    path = Path(checkpoint_path)
    if path.exists():
        path.unlink()