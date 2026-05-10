"""LangGraph agent: subtitle translation state machine with resume support."""

import logging
from typing import TypedDict
from langgraph.graph import StateGraph, END

from src.parser import parse_srt, write_srt
from src.checkpoint import load_checkpoint, save_checkpoint, delete_checkpoint
from src.translator import translate_batch
from src.validator import translate_with_retry
from src.prompts import build_system_prompt, BATCH_PROMPT

logger = logging.getLogger(__name__)

BATCH_SIZE = 50


class TranslationState(TypedDict):
    """State shared across all LangGraph nodes."""
    input_path: str
    output_path: str
    source_lang: str
    target_lang: str
    entries: list
    current_index: int
    translated: list[str]
    batch_size: int
    errors: list[str]
    done: bool


def node_parse(state: TranslationState) -> TranslationState:
    """Node 1: Parse the .srt file into entries."""
    logger.info("Parsing SRT: %s", state["input_path"])
    entries = parse_srt(state["input_path"])
    logger.info("Found %d subtitle entries", len(entries))
    state["entries"] = entries
    state["current_index"] = 0
    state["translated"] = []
    state["errors"] = []
    state["done"] = False
    return state


def node_checkpoint(state: TranslationState) -> TranslationState:
    """Node 2: Load checkpoint if exists, resume from last saved index."""
    ckpt = load_checkpoint(state["input_path"])
    if ckpt and ckpt.get("current_index", 0) > 0:
        resume_idx = ckpt["current_index"]
        logger.info(
            "Checkpoint found — resuming from entry %d of %d",
            resume_idx,
            ckpt.get("total_entries", "?"),
        )
        state["current_index"] = resume_idx
        state["translated"] = ckpt.get("translated", [])
    else:
        logger.info("No checkpoint — starting fresh translation")
        state["current_index"] = 0
        state["translated"] = []
    return state


def node_translate_batch(state: TranslationState) -> TranslationState:
    """Node 3: Translate one batch of BATCH_SIZE entries."""
    entries = state["entries"]
    idx = state["current_index"]
    batch_size = state.get("batch_size", BATCH_SIZE)

    if idx >= len(entries):
        state["done"] = True
        return state

    batch_end = min(idx + batch_size, len(entries))
    batch = entries[idx:batch_end]
    batch_texts = [e["content"] for e in batch]

    logger.info(
        "Translating batch: entries %d-%d of %d",
        idx + 1,
        batch_end,
        len(entries),
    )

    system_prompt = build_system_prompt(
        state["source_lang"], state["target_lang"]
    )

    try:
        translations = translate_with_retry(
            translate_batch,
            batch_texts,
            system_prompt,
            BATCH_PROMPT,
        )
    except Exception as e:
        logger.error("Batch translation failed: %s", str(e))
        state["errors"].append(f"Failed at entry {idx + 1}: {str(e)}")
        raise

    state["translated"].extend(translations)
    state["current_index"] = batch_end
    return state


def node_save_checkpoint(state: TranslationState) -> TranslationState:
    """Node 4: Save progress checkpoint after each successful batch."""
    save_checkpoint(
        state["input_path"],
        state["current_index"],
        len(state["entries"]),
        state["translated"],
    )
    logger.info(
        "Checkpoint saved — %d/%d entries translated",
        state["current_index"],
        len(state["entries"]),
    )
    return state


def node_should_continue(state: TranslationState) -> str:
    """Router: continue translating or finish?"""
    if state["done"] or state["current_index"] >= len(state["entries"]):
        return "write"
    return "translate"


def node_write(state: TranslationState) -> TranslationState:
    """Node 5: Write the translated .srt file and clean up checkpoint."""
    entries = state["entries"]
    translated = state["translated"]

    output_entries = []
    for i, entry in enumerate(entries):
        target_text = translated[i] if i < len(translated) else entry["content"]
        output_entries.append({
            "index": entry["index"],
            "start": entry["start"],
            "end": entry["end"],
            "content": target_text,
        })

    write_srt(state["output_path"], output_entries, state["target_lang"])
    logger.info("Wrote translated SRT: %s", state["output_path"])

    delete_checkpoint(state["input_path"])
    logger.info("Checkpoint deleted — translation complete")

    return state


def build_graph() -> StateGraph:
    """Build the LangGraph state machine."""
    graph = StateGraph(TranslationState)

    graph.add_node("parse", node_parse)
    graph.add_node("checkpoint", node_checkpoint)
    graph.add_node("translate", node_translate_batch)
    graph.add_node("save_ckpt", node_save_checkpoint)
    graph.add_node("write", node_write)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "checkpoint")
    graph.add_edge("checkpoint", "translate")
    graph.add_edge("translate", "save_ckpt")
    graph.add_conditional_edges(
        "save_ckpt",
        node_should_continue,
        {
            "translate": "translate",
            "write": "write",
        },
    )
    graph.add_edge("write", END)

    return graph


def run_translation(
    input_path: str,
    output_path: str,
    source_lang: str = "en",
    target_lang: str = "ar",
) -> None:
    """Run the full translation pipeline.

    Args:
        input_path: Path to source .srt file (e.g., input/TheMatrix.srt).
        output_path: Path for translated .srt output (e.g., output/TheMatrix.ar.srt).
        source_lang: Source language code (default: 'en').
        target_lang: Target language code (default: 'ar').
    """
    graph = build_graph()
    app = graph.compile()

    initial_state: TranslationState = {
        "input_path": input_path,
        "output_path": output_path,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "entries": [],
        "current_index": 0,
        "translated": [],
        "batch_size": BATCH_SIZE,
        "errors": [],
        "done": False,
    }

    logger.info(
        "Starting translation [%s → %s]: %s -> %s",
        source_lang,
        target_lang,
        input_path,
        output_path,
    )
    final_state = app.invoke(initial_state)

    total = len(final_state["entries"])
    errors = final_state.get("errors", [])
    if errors:
        logger.warning("Completed with %d errors", len(errors))
        for err in errors:
            logger.warning("  - %s", err)
    else:
        logger.info("Translation complete — %d entries", total)