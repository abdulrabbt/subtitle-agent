# Subtitle Translation Agent — Architecture Plan

## Overview

A **LangGraph-based AI agent** that translates English `.srt` subtitle files to Arabic using **DeepSeek V4 Pro**. It preserves timestamps, supports **resume-from-failure**, and produces movie/TV-appropriate translations. Filenames use the movie/show name (e.g., `TheMatrix.srt` → `TheMatrix.ar.srt`).

## Tech Stack

| Component     | Choice                    |
| ------------- | ------------------------- |
| Language      | Python 3.10+              |
| Orchestration | LangGraph                 |
| LLM Provider  | DeepSeek V4 Pro           |
| API Library   | `openai` (compatible API) |
| SRT Parsing   | `srt` library             |
| Checkpoints   | JSON file per input       |
| Config        | `.env` file               |

## File Structure

```
subtitle-agent/
├── .env                  # DEEPSEEK_API_KEY=sk-... (user fills manually)
├── .env.example          # Template
├── requirements.txt      # langgraph, openai, srt, python-dotenv, pydantic
├── main.py               # CLI: python main.py TheMatrix.srt TheMatrix.ar.srt
├── README.md             # This file — architecture documentation
└── src/
    ├── __init__.py        # Package marker
    ├── prompts.py         # Translation prompt for movie/TV Arabic
    ├── parser.py          # SRT parse/write — timestamps NEVER touched
    ├── checkpoint.py      # JSON checkpoint read/write for resume support
    ├── translator.py      # DeepSeek API client (OpenAI-compatible)
    ├── validator.py       # Response validation + 3x retry logic
    └── agent.py           # LangGraph state machine (5 nodes + router)
```

## LangGraph Workflow

```
[Parse] → [Checkpoint] → [Translate] → [Save Ckpt] → [Write]
                ↑                          │
                └── resume on restart ─────┘
                          (loop)
```

### Graph Nodes (5 + conditional router)

| #   | Node Name         | Function                                     | Module                           |
| --- | ----------------- | -------------------------------------------- | -------------------------------- |
| 1   | `parse`           | Read .srt → list of entries                  | `parser.py`                      |
| 2   | `checkpoint`      | Load JSON checkpoint (if any)                | `checkpoint.py`                  |
| 3   | `translate`       | Translate batch + validate + retry           | `translator.py` + `validator.py` |
| 4   | `save_ckpt`       | Save JSON checkpoint                         | `checkpoint.py`                  |
| 5   | `write`           | Compose .srt, delete checkpoint              | `parser.py` + `checkpoint.py`    |
| —   | `should_continue` | Router: `translate` if more, `write` if done | (inline in agent.py)             |

### State Definition (matches `src/agent.py`)

```python
class TranslationState(TypedDict):
    input_path: str       # Source .srt file path
    output_path: str      # Destination .srt file path
    entries: list         # Parsed SRT entries [{index, start, end, content}]
    current_index: int    # Resume point (0-based)
    translated: list[str] # Completed Arabic translations
    batch_size: int       # Entries per LLM call (default: 50)
    errors: list[str]     # Error log
    done: bool            # Translation complete flag
```

## Translation Flow (per batch)

```
Translate Batch (50 entries)
    │
    ├─ Call DeepSeek API (deepseek-v4-flash or deepseek-v4-pro via DEEPSEEK_MODEL)
    ├─ Validate response (line count == 50)
    │   ├─ Valid → save to state.translated
    │   └─ Invalid → retry (up to 3 attempts)
    └─ On exception → save checkpoint, raise error for resume
```

## Quality Assurance

| Layer                      | Status         | Description                                                     |
| -------------------------- | -------------- | --------------------------------------------------------------- |
| **Timestamp Preservation** | ✅ Implemented | `srt` library handles timestamps — only `.content` is replaced  |
| **Response Validation**    | ✅ Implemented | Checks line count matches batch size; 3x auto-retry on mismatch |
| **LLM Self-Review**        | 🔜 Planned     | Optional second-pass QA for natural Arabic & cultural fit       |

## Resume Logic

1. Checkpoint saved as `{MovieName}.srt.checkpoint.json` after every successful batch
2. On restart, `node_checkpoint` checks if checkpoint exists → loads `current_index` and `translated`
3. Translation resumes from `current_index`
4. On successful completion, checkpoint is **deleted**
5. If interrupted (Ctrl+C), checkpoint remains for next run

### Checkpoint JSON Example

```json
{
  "input_path": "TheMatrix.srt",
  "current_index": 34,
  "total_entries": 500,
  "translated": ["مرحباً", "كيف حالك؟", "..."],
  "last_updated": "2026-05-06T20:30:00"
}
```

## CLI Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
cp .env.example .env
# Edit .env: DEEPSEEK_API_KEY=sk-your-real-key

# Translate (movie name = filename)
python main.py TheMatrix.srt TheMatrix.ar.srt

# If it fails at entry 150, just re-run — auto-resumes!
python main.py TheMatrix.srt TheMatrix.ar.srt
```

## Dependencies (`requirements.txt`)

```
langgraph>=0.2.0
openai>=1.0.0
srt>=3.5.0
python-dotenv>=1.0.0
pydantic>=2.0.0
```
