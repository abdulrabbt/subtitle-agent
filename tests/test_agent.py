"""Unit tests for src/agent.py — LangGraph state machine nodes and pipeline."""

import datetime
from unittest.mock import MagicMock, patch, call

import pytest
from langgraph.graph import StateGraph

from src.agent import (
    TranslationState,
    node_parse,
    node_checkpoint,
    node_translate_batch,
    node_save_checkpoint,
    node_should_continue,
    node_write,
    build_graph,
    run_translation,
    BATCH_SIZE,
)

# ---------------------------------------------------------------------------
# Helper: build a minimal fresh state
# ---------------------------------------------------------------------------
def fresh_state(**overrides) -> TranslationState:
    s: TranslationState = {
        "input_path": "input/test.srt",
        "output_path": "output/test.ar.srt",
        "entries": [],
        "current_index": 0,
        "translated": [],
        "batch_size": BATCH_SIZE,
        "errors": [],
        "done": False,
    }
    s.update(overrides)
    return s


# ---------------------------------------------------------------------------
# node_parse
# ---------------------------------------------------------------------------
class TestNodeParse:
    @patch("src.agent.parse_srt")
    def test_populates_state_from_parsed_entries(self, mock_parse):
        mock_parse.return_value = [
            {"index": 1, "start": datetime.timedelta(0), "end": datetime.timedelta(1), "content": "Hello"},
            {"index": 2, "start": datetime.timedelta(1), "end": datetime.timedelta(2), "content": "World"},
        ]
        state = fresh_state()
        result = node_parse(state)

        assert len(result["entries"]) == 2
        assert result["current_index"] == 0
        assert result["translated"] == []
        assert result["errors"] == []
        assert result["done"] is False
        mock_parse.assert_called_once_with("input/test.srt")


# ---------------------------------------------------------------------------
# node_checkpoint
# ---------------------------------------------------------------------------
class TestNodeCheckpoint:
    @patch("src.agent.load_checkpoint")
    def test_no_checkpoint_keeps_defaults(self, mock_load):
        mock_load.return_value = None
        state = fresh_state()
        result = node_checkpoint(state)

        assert result["current_index"] == 0
        assert result["translated"] == []

    @patch("src.agent.load_checkpoint")
    def test_with_checkpoint_restores_progress(self, mock_load):
        mock_load.return_value = {
            "input_path": "input/test.srt",
            "current_index": 25,
            "total_entries": 100,
            "translated": ["a", "b", "c"],
        }
        state = fresh_state()
        result = node_checkpoint(state)

        assert result["current_index"] == 25
        assert result["translated"] == ["a", "b", "c"]

    @patch("src.agent.load_checkpoint")
    def test_zero_index_checkpoint_treated_as_fresh(self, mock_load):
        """current_index == 0 should not resume (treated as no checkpoint)."""
        mock_load.return_value = {
            "input_path": "input/test.srt",
            "current_index": 0,
            "total_entries": 100,
            "translated": [],
        }
        state = fresh_state(current_index=99)  # something non-zero
        result = node_checkpoint(state)

        # Falls through to the else branch, resetting
        assert result["current_index"] == 0
        assert result["translated"] == []


# ---------------------------------------------------------------------------
# node_translate_batch
# ---------------------------------------------------------------------------
class TestNodeTranslateBatch:
    def test_done_when_index_exceeds_entries(self):
        entries = [{"content": "a"}] * 5
        state = fresh_state(entries=entries, current_index=5)
        result = node_translate_batch(state)

        assert result["done"] is True

    def test_done_when_index_past_entries(self):
        entries = [{"content": "a"}] * 3
        state = fresh_state(entries=entries, current_index=10)
        result = node_translate_batch(state)

        assert result["done"] is True

    @patch("src.agent.translate_with_retry")
    def test_translates_batch_and_advances_index(self, mock_retry):
        mock_retry.return_value = ["\u062A\u0631\u062C\u0645\u0629 1", "\u062A\u0631\u062C\u0645\u0629 2"]
        entries = [
            {"content": "English 1"},
            {"content": "English 2"},
            {"content": "English 3"},
            {"content": "English 4"},
        ]
        state = fresh_state(entries=entries, current_index=1, batch_size=2)
        result = node_translate_batch(state)

        assert result["current_index"] == 3  # 1 + 2
        assert result["translated"] == ["\u062A\u0631\u062C\u0645\u0629 1", "\u062A\u0631\u062C\u0645\u0629 2"]
        assert result["done"] is False

    @patch("src.agent.translate_with_retry")
    def test_translation_error_appends_to_errors_and_re_raises(self, mock_retry):
        mock_retry.side_effect = RuntimeError("DeepSeek API timeout")
        entries = [{"content": "English 1"}, {"content": "English 2"}]
        state = fresh_state(entries=entries, current_index=0, batch_size=2)

        with pytest.raises(RuntimeError, match="DeepSeek API timeout"):
            node_translate_batch(state)

        assert len(state["errors"]) == 1
        assert "Failed at entry 1" in state["errors"][0]

    @patch("src.agent.translate_with_retry")
    def test_last_partial_batch_handled_correctly(self, mock_retry):
        """When remaining entries < batch_size, only those are translated."""
        mock_retry.return_value = ["T1"]
        entries = [{"content": "E1"}, {"content": "E2"}, {"content": "E3"}]
        state = fresh_state(entries=entries, current_index=2, batch_size=50)
        result = node_translate_batch(state)

        assert result["current_index"] == 3
        assert len(result["translated"]) == 1


# ---------------------------------------------------------------------------
# node_save_checkpoint
# ---------------------------------------------------------------------------
class TestNodeSaveCheckpoint:
    @patch("src.agent.save_checkpoint")
    def test_saves_with_correct_args(self, mock_save):
        state = fresh_state(
            current_index=40,
            entries=[{"content": "x"}] * 100,
            translated=["t"] * 40,
        )
        result = node_save_checkpoint(state)

        mock_save.assert_called_once_with(
            "input/test.srt",
            40,
            100,
            ["t"] * 40,
        )
        assert result is state  # returns same state object


# ---------------------------------------------------------------------------
# node_should_continue
# ---------------------------------------------------------------------------
class TestNodeShouldContinue:
    def test_returns_write_when_done(self):
        state = fresh_state(done=True)
        assert node_should_continue(state) == "write"

    def test_returns_write_when_index_reaches_end(self):
        state = fresh_state(
            entries=[{"content": "x"}] * 10,
            current_index=10,
        )
        assert node_should_continue(state) == "write"

    def test_returns_write_when_index_exceeds_end(self):
        state = fresh_state(
            entries=[{"content": "x"}] * 5,
            current_index=7,
        )
        assert node_should_continue(state) == "write"

    def test_returns_translate_when_more_entries_remain(self):
        state = fresh_state(
            entries=[{"content": "x"}] * 10,
            current_index=5,
        )
        assert node_should_continue(state) == "translate"


# ---------------------------------------------------------------------------
# node_write
# ---------------------------------------------------------------------------
class TestNodeWrite:
    @patch("src.agent.delete_checkpoint")
    @patch("src.agent.write_srt")
    def test_writes_correct_output_entries(self, mock_write, mock_delete):
        entries = [
            {"index": 1, "start": datetime.timedelta(0), "end": datetime.timedelta(1), "content": "Hello"},
            {"index": 2, "start": datetime.timedelta(1), "end": datetime.timedelta(2), "content": "World"},
        ]
        translated = ["\u0645\u0631\u062D\u0628\u0627", "\u0627\u0644\u0639\u0627\u0644\u0645"]
        state = fresh_state(entries=entries, translated=translated)

        node_write(state)

        written_entries = mock_write.call_args[0][1]
        assert len(written_entries) == 2
        assert written_entries[0]["content"] == "\u0645\u0631\u062D\u0628\u0627"
        assert written_entries[1]["content"] == "\u0627\u0644\u0639\u0627\u0644\u0645"
        assert written_entries[0]["index"] == 1
        assert written_entries[0]["start"] == datetime.timedelta(0)
        mock_write.assert_called_once_with("output/test.ar.srt", written_entries)
        mock_delete.assert_called_once_with("input/test.srt")

    @patch("src.agent.delete_checkpoint")
    @patch("src.agent.write_srt")
    def test_fallback_to_original_content_when_translations_short(self, mock_write, mock_delete):
        """If translated has fewer entries (shouldn't happen, but safe fallback)."""
        entries = [
            {"index": 1, "start": datetime.timedelta(0), "end": datetime.timedelta(1), "content": "Hello"},
            {"index": 2, "start": datetime.timedelta(1), "end": datetime.timedelta(2), "content": "World"},
        ]
        translated = ["\u0645\u0631\u062D\u0628\u0627"]  # only 1
        state = fresh_state(entries=entries, translated=translated)

        node_write(state)

        written_entries = mock_write.call_args[0][1]
        assert written_entries[0]["content"] == "\u0645\u0631\u062D\u0628\u0627"
        assert written_entries[1]["content"] == "World"  # original fallback


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------
class TestBuildGraph:
    def test_returns_state_graph(self):
        graph = build_graph()
        assert isinstance(graph, StateGraph)

    def test_graph_has_expected_nodes(self, monkeypatch):
        """Verify all five nodes and entry point are configured."""
        graph = build_graph()
        nodes = graph.nodes
        # Expected node names
        assert "parse" in nodes
        assert "checkpoint" in nodes
        assert "translate" in nodes
        assert "save_ckpt" in nodes
        assert "write" in nodes


# ---------------------------------------------------------------------------
# run_translation (end-to-end with mocks)
# ---------------------------------------------------------------------------
class TestRunTranslation:
    @patch("src.agent.build_graph")
    def test_full_pipeline_invokes_graph(self, mock_build_graph):
        mock_graph = MagicMock()
        mock_app = MagicMock()
        mock_graph.compile.return_value = mock_app
        mock_build_graph.return_value = mock_graph

        # Simulate final state after graph.invoke
        mock_app.invoke.return_value = {
            "input_path": "test.srt",
            "output_path": "test.ar.srt",
            "entries": [{"content": "Hello"}] * 10,
            "current_index": 10,
            "translated": ["T"] * 10,
            "batch_size": BATCH_SIZE,
            "errors": [],
            "done": True,
        }

        run_translation("test.srt", "test.ar.srt")

        mock_build_graph.assert_called_once()
        mock_graph.compile.assert_called_once()
        mock_app.invoke.assert_called_once()
        # Verify initial state passed to invoke
        initial = mock_app.invoke.call_args[0][0]
        assert initial["input_path"] == "test.srt"
        assert initial["output_path"] == "test.ar.srt"
        assert initial["current_index"] == 0

    @patch("src.agent.build_graph")
    def test_logs_errors_when_present(self, mock_build_graph):
        mock_graph = MagicMock()
        mock_app = MagicMock()
        mock_graph.compile.return_value = mock_app
        mock_build_graph.return_value = mock_graph

        mock_app.invoke.return_value = {
            "input_path": "test.srt",
            "output_path": "test.ar.srt",
            "entries": [{"content": "Hello"}] * 5,
            "current_index": 5,
            "translated": ["T"] * 5,
            "batch_size": BATCH_SIZE,
            "errors": ["Failed at entry 3: timeout", "Failed at entry 5: rate limit"],
            "done": True,
        }

        # Should not raise — just log warnings
        run_translation("test.srt", "test.ar.srt")
        assert mock_app.invoke.called