# Subtitle Translation Agent — Architecture Plan

## Overview

A **LangGraph-based AI agent** that translates `.srt` subtitle files using **any AI model (OpenAI-compatible API)**. Supports 23+ language pairs (English, Arabic, Chinese, Japanese, Korean, French, German, Spanish, Portuguese, Russian, Italian, Turkish, Hindi, Thai, Vietnamese, Bengali, Persian, Hebrew, Urdu, Pashto, Sindhi, Uyghur, Divehi, Yiddish, Kurdish). Preserves timestamps, supports **resume-from-failure**, and produces movie/TV-appropriate translations with automatic RTL handling.

## Tech Stack

| Component     | Choice                    |
| ------------- | ------------------------- |
| Language      | Python 3.10+              |
| Orchestration | LangGraph                 |
| LLM Provider  | any AI model (OpenAI-compatible API)           |
| API Library   | `openai` (compatible API) |
| SRT Parsing   | `srt` library             |
| Checkpoints   | JSON file per input       |
| Config        | `.env` file               |

## File Structure

```
subtitle-agent/
├── .env                  # DEEPSEEK_API_KEY=sk-... (user fills manually)
├── .env.example          # Template
├── requirements.txt      # langgraph, openai, srt, python-dotenv, pydantic, pytest
├── main.py               # CLI: python main.py <input.srt> <output.srt> [--source-lang CODE] [--target-lang CODE] [--debug]
├── README.md             # This file — architecture documentation
├── src/
│   ├── __init__.py        # Package marker
│   ├── prompts.py         # Dynamic system prompts for 23+ language pairs with RTL support
│   ├── parser.py          # SRT parse/write — timestamps NEVER touched
│   ├── checkpoint.py      # JSON checkpoint read/write for resume support
│   ├── translator.py      # DeepSeek API client (OpenAI-compatible)
│   ├── validator.py       # Response validation + fuzzy recovery + 3x retry
│   └── agent.py           # LangGraph state machine (5 nodes + router)
└── tests/
    ├── __init__.py
    ├── test_agent.py
    ├── test_checkpoint.py
    ├── test_parser.py
    ├── test_prompts.py
    ├── test_translator.py
    └── test_validator.py
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
    source_lang: str      # Source language code (e.g., 'en')
    target_lang: str      # Target language code (e.g., 'ar')
    entries: list         # Parsed SRT entries [{index, start, end, content}]
    current_index: int    # Resume point (0-based)
    translated: list[str] # Completed translations
    batch_size: int       # Entries per LLM call (default: 50)
    errors: list[str]     # Error log
    done: bool            # Translation complete flag
```

## Translation Flow (per batch)

```
Translate Batch (50 entries)
    │
    ├─ Call DeepSeek API (deepseek-v4-flash default; configurable via DEEPSEEK_MODEL)
    ├─ Validate response (line count == 50)
    │   ├─ Valid → save to state.translated
    │   ├─ Extra lines → fuzzy recovery: take first 50, discard extras
    │   └─ Too few lines → retry (up to 3 attempts)
    └─ On exception → save checkpoint, raise error for resume
```

## Quality Assurance

| Layer                      | Status         | Description                                                                                                                                                                      |
| -------------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp Preservation** | ✅ Implemented | `srt` library handles timestamps — only `.content` is replaced                                                                                                                   |
| **RTL Formatting**         | ✅ Implemented | RTL text (Arabic, Persian, Hebrew, Urdu, etc.) is automatically wrapped in Unicode RLE/PDF markers (`\u202B`...`\u202C`) for correct bidirectional rendering in subtitle players |
| **Response Validation**    | ✅ Implemented | Checks line count matches batch size; fuzzy recovery for extra lines; 3x auto-retry on mismatch                                                                                  |
| **LLM Self-Review**        | 🔜 Planned     | Optional second-pass QA for naturalness & cultural fit across all target languages                                                                                               |

## Environment Variables

Set via `.env` file (copy from `.env.example`):

| Variable            | Required | Default                       | Description                          |
| ------------------- | -------- | ----------------------------- | ------------------------------------ |
| `DEEPSEEK_API_KEY`  | ✅ Yes   | —                             | DeepSeek API key                     |
| `DEEPSEEK_BASE_URL` | No       | `https://api.deepseek.com/v1` | DeepSeek API endpoint                |
| `DEEPSEEK_MODEL`    | No       | `deepseek-v4-flash`           | Model name (e.g., `deepseek-v4-pro`) |
| `SOURCE_LANG`       | No       | `en`                          | Default source language code         |
| `TARGET_LANG`       | No       | `ar`                          | Default target language code         |

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

# Translate English to Arabic (default)
python main.py TheMatrix.srt TheMatrix.ar.srt

# Translate to another language (23+ supported)
python main.py TheMatrix.srt TheMatrix.fr.srt --target-lang fr

# Specify both source and target languages
python main.py TheMatrix.srt TheMatrix.es.srt --source-lang en --target-lang es

# Enable debug mode for verbose logging (API calls, raw responses, validation details)
python main.py TheMatrix.srt TheMatrix.ar.srt --debug

# If it fails at entry 150, just re-run — auto-resumes!
python main.py TheMatrix.srt TheMatrix.ar.srt
```

> **Note:** `main.py` automatically creates parent directories for input/output paths if they don't exist.

## Dependencies (`requirements.txt`)

```
langgraph>=0.2.0
openai>=1.0.0
srt>=3.5.0
python-dotenv>=1.0.0
pydantic>=2.0.0
pytest>=8.0.0
```
