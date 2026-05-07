"""Unit tests for src/checkpoint.py."""

import tempfile
from pathlib import Path

from src.checkpoint import (
    get_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
    delete_checkpoint,
)


# ---------------------------------------------------------------------------
# get_checkpoint_path
# ---------------------------------------------------------------------------
class TestGetCheckpointPath:
    def test_appends_suffix_correctly(self):
        result = get_checkpoint_path("/some/path/TheMatrix.srt")
        assert result == "/some/path/TheMatrix.srt.checkpoint.json"

    def test_handles_relative_path(self):
        result = get_checkpoint_path("input/TheMatrix.srt")
        assert result == "input/TheMatrix.srt.checkpoint.json"


# ---------------------------------------------------------------------------
# load_checkpoint
# ---------------------------------------------------------------------------
class TestLoadCheckpoint:
    def test_returns_none_when_no_file_exists(self):
        result = load_checkpoint("nonexistent.srt")
        assert result is None

    def test_returns_none_when_input_path_mismatch(self):
        """Checkpoint exists but input_path field doesn't match — should return None."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", encoding="utf-8", delete=False
        ) as f:
            input_path = f.name
            ckpt_path = get_checkpoint_path(input_path)

        try:
            # Write a checkpoint with a non-matching input_path
            import json
            data = {
                "input_path": "different/path.srt",
                "current_index": 5,
                "total_entries": 100,
                "translated": ["a", "b"],
            }
            Path(ckpt_path).write_text(json.dumps(data), encoding="utf-8")
            result = load_checkpoint(input_path)
            assert result is None
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(ckpt_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint roundtrip
# ---------------------------------------------------------------------------
class TestSaveAndLoadCheckpoint:
    def test_roundtrip_preserves_all_fields(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", encoding="utf-8", delete=False
        ) as f:
            input_path = f.name

        ckpt_path = get_checkpoint_path(input_path)
        try:
            translated = ["\u0645\u0631\u062d\u0628\u0627", "\u0643\u064a\u0641 \u062d\u0627\u0644\u0643"]
            save_checkpoint(input_path, 10, 100, translated[:1])
            data = load_checkpoint(input_path)
            assert data is not None
            assert data["input_path"] == input_path
            assert data["current_index"] == 10
            assert data["total_entries"] == 100
            assert data["translated"] == translated[:1]
            assert "last_updated" in data

            # Save again with more progress
            save_checkpoint(input_path, 50, 100, translated)
            data = load_checkpoint(input_path)
            assert data["current_index"] == 50
            assert data["translated"] == translated
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(ckpt_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# delete_checkpoint
# ---------------------------------------------------------------------------
class TestDeleteCheckpoint:
    def test_delete_removes_checkpoint_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", encoding="utf-8", delete=False
        ) as f:
            input_path = f.name

        ckpt_path = get_checkpoint_path(input_path)
        try:
            save_checkpoint(input_path, 5, 50, ["a", "b", "c", "d", "e"])
            assert Path(ckpt_path).exists()
            delete_checkpoint(input_path)
            assert not Path(ckpt_path).exists()
        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(ckpt_path).unlink(missing_ok=True)

    def test_delete_nonexistent_does_not_raise(self):
        """Deleting a checkpoint that doesn't exist should not error."""
        delete_checkpoint("nonexistent_file_xyz.srt")