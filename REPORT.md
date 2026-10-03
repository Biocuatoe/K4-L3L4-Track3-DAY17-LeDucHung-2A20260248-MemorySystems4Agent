# REPORT.md — Day 17: Memory Systems for AI Agent

## Overview

This report documents the full implementation of the Day 17 educational lab "Memory Systems for AI Agent". All seven student scaffold files in `src/` have been implemented, tested, and benchmarked. The implementation satisfies the requirements in `Guide.md` and `Rubric.md`, operates in offline mode (no external dependencies required), and achieves measurable improvements in cross-session recall and prompt token efficiency on the stress benchmark.

---

## Implementation Summary

### Files Modified / Created

| File | Description |
|---|---|
| `src/model_provider.py` | `ProviderConfig` dataclass, `normalize_provider()` (6 provider aliases), `build_chat_model()` with lazy imports |
| `src/config.py` | `LabConfig` dataclass, `load_config()` with `.env` support, state dir init |
| `src/memory_store.py` | `estimate_tokens()`, `UserProfileStore` (read/write/edit/facts/upsert), `extract_profile_updates()`, `summarize_messages()`, `CompactMemoryManager` |
| `src/agent_baseline.py` | `BaselineAgent` — within-thread memory only, no persistent storage |
| `src/agent_advanced.py` | `AdvancedAgent` — 3-layer memory (short-term + persistent User.md + compact), offline response from profile facts |
| `src/benchmark.py` | `load_conversations()`, `recall_points()`, `heuristic_quality()`, `run_agent_benchmark()`, `format_rows()` with tabulate fallback, `main()` |
| `src/test_agents.py` | Pytest suite: 4 core tests + 14 edge case tests, all using `tmp_path` fixture |
| `REPORT.md` | This document |

### Architecture

```
User message
    │
    ▼
extract_profile_updates()        ← NLP extraction (Vietnamese + English)
    │                           Strict patterns + post-filtering for stability
    ▼
profile_store.upsert_fact()    ← Persistent User.md (field-level upsert)
    │
    ▼
compact_memory.append()         ← Within-thread short-term memory
    │                           Auto-compacts when threshold exceeded
    ▼
_offline_response()            ← Fact-aware deterministic response
    │
    ▼
Return { answer, prompt_tokens, agent_tokens, memory_path, compaction_count }
```

---

## Benchmark Results

### Standard Benchmark (`data/conversations.json`, 10 conversations)

| Agent | Agent tokens | Prompt tokens | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 1,452 | 13,964 | 0.10 | 0.51 | 0 | 0 |
| Advanced | 2,177 | 24,384 | 0.14 | 0.51 | 439 | 0 |

### Long-Context Stress Benchmark (`data/advanced_long_context.json`, 16 turns)

| Agent | Agent tokens | Prompt tokens | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 239 | 18,006 | 0.00 | 0.46 | 0 | 0 |
| Advanced | 190 | 8,083 | 0.42 | 0.65 | 415 | 5 |

---

## Analysis

### Why Advanced Outperforms Baseline on the Stress Benchmark

1. **Cross-session recall (0.42 vs 0.00):** Baseline forgets all facts when switching thread IDs. Advanced persists facts in `User.md`, enabling it to answer recall questions in new threads. Baseline achieves 0.00 because it has no persistent storage — it starts each new thread with an empty state.

2. **Prompt token reduction (>55%):** Baseline accumulates all raw messages in its session. On the 16-turn stress test, the cumulative prompt token cost grows to 18,006. Advanced compacts the thread 5 times, replacing older messages with a compressed summary. The compact representation (User.md + summary + recent messages) totals only 8,083 prompt tokens — a **54.5% reduction**.

3. **Memory growth (415 bytes):** Advanced writes a structured `User.md` profile that accumulates stable facts across turns. Baseline creates no files.

4. **Response quality (0.65 vs 0.46):** Advanced's responses incorporate known profile facts, making them more relevant and informative. Baseline's generic acknowledgements score lower.

5. **No compactions on Standard Benchmark:** The standard conversations (~10 turns each) stay well under the 800-token threshold, so no compaction occurs. This is expected — compaction only triggers on genuinely long threads.

---

## Bonus Features Implemented

### 1. Confidence Filtering

**Problem:** A naive extractor would store everything a user says, including questions, jokes, and hypothetical statements.

**Implementation:** `extract_profile_updates()` applies multiple layers of filtering:
- **Question detection:** Skip messages ending in `?` unless an assertion verb is present
- **Hypothetical signals:** Skip messages containing "đùa", "joke", "what if", "hay là"
- **Temporary location signals:** Skip "đi họp ở", "bay ra Hà Nội họp", "họp tại" — these are meeting locations, not residence
- **Correction-aware filtering:** If a correction signal ("thực ra", "không còn", "giờ là") appears, only the NEW location after the correction is stored

**Measured effect:** On the stress test, Advanced correctly updates location from Huế → Đà Nẵng and ignores both the temporary "Hà Nội" meeting location and the joking "product manager" profession.

### 2. Entity Extraction (Field-level Structured Facts)

**Problem:** Extracting raw conversational text and storing it verbatim would pollute the profile.

**Implementation:** Each field has a dedicated regex pattern with:
- **Strict anchor requirements:** The pattern only fires when a specific introduction keyword is present (e.g., "tên là" for name, "đang làm" for profession)
- **Character class restriction:** Captured values use `[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9\s]*` — not arbitrary text
- **Lookahead/lookbehind:** Patterns stop at conjunction words, question marks, and explanatory particles
- **Post-processing:** Additional filters remove conversational fragments, reaffirmation noise ("vẫn giữ nguyên"), and cross-field contamination (e.g., a response style fragment being stored as favorite food)

**Example:**
- Input: `"Mình thích trả lời ngắn gọn thành 3 bullet"`
- Captured: `response style: ngắn gọn thành 3 bullet` (clean, structured)
- NOT captured: conversational text like "trade-off giữa recall và token cost"

### 3. Conflict-Aware Memory Model

**Problem:** The profile must handle corrections gracefully — if a user says "I was at Huế, but now I'm at Đà Nẵng", only Đà Nẵng should remain.

**Implementation:**
- `UserProfileStore.upsert()` performs field-level replacement: if a field already exists, its value is replaced rather than duplicated
- `extract_profile_updates()` detects correction signals and skips old location values that appear before the correction marker
- The upsert uses a `re.subn` replacement that preserves the markdown structure (label prefix remains, only the value changes)

**Measured effect:** In the stress test, the profile correctly shows `Location: Đà Nẵng` (not "Huế") and `Response style: ngắn gọn theo 3 bullet có ví dụ thực chiến` (the final preference, not an earlier one).

### 4. Heuristic Token Estimation

**Problem:** Exact tokenizers (tiktoken, sentencepiece) require external packages.

**Implementation:** `estimate_tokens(text)` = `max(1, round(non_whitespace_chars / 4))`. This heuristic is:
- Deterministic (same input always gives same output)
- Fast (O(n) string scan)
- Provider-agnostic (works offline)
- Used consistently throughout: CompactMemoryManager threshold comparison, prompt token accumulation, token usage reporting

---

## Known Limitations

1. **NLP extraction recall is heuristic:** The regex-based extractor may miss facts expressed in unusual phrasing. A production system would use an LLM for extraction. The benchmark's 0.42 recall score reflects this — the extractor successfully captures name, location, and response style but is less reliable for profession and technical interests in conversational Vietnamese text.

2. **Cross-session recall on Standard Benchmark is low (0.14):** The standard conversations contain fewer explicit self-introductions and preference statements. The extractor is conservative by design to avoid false positives, which means it also misses some legitimate facts.

3. **Response quality scores are moderate:** The offline deterministic response generator uses simple keyword matching. A live LLM would produce richer, more contextually appropriate answers.

4. **Correction detection is location-specific:** The correction-aware filter currently only applies to the `location` field. Extending it to other fields (profession corrections, name corrections) would require additional signal patterns.

---

## Test Coverage

All 18 pytest tests pass, covering:

- `test_user_markdown_read_write_edit` — UserProfileStore CRUD operations
- `test_compact_trigger` — CompactMemoryManager triggers compaction above threshold
- `test_cross_session_recall` — Advanced recalls facts across threads; Baseline does not
- `test_compact_reduces_prompt_load_on_long_thread` — Compacted prompt < raw message total
- 14 edge cases: empty messages, missing profiles, missing fields, fact deduplication, correction replacement, question-only turns, temporary locations, jokes, path traversal, Unicode Vietnamese text, multiple compactions, very long messages, zero-context threads, no persistent file for Baseline

---

## Reference: Key Implementation Details

### `src/memory_store.py`

- **`estimate_tokens`**: `max(1, round(len(re.sub(r"\s+", "", text)) / 4))`
- **`UserProfileStore.upsert`**: NFKC path sanitization, case-insensitive field label matching, markdown-structured storage
- **`extract_profile_updates`**: 8 field patterns, 4 signal filters (question, hypothetical, temporary, correction), post-processing for conversational fragments
- **`CompactMemoryManager`**: Threshold-based compaction, summary merging, `keep_messages` recent history preservation

### `src/agent_advanced.py`

- **`_reply_offline`**: Extract → Persist → Compact → Estimate → Respond → Accumulate
- **`_offline_response`**: Intent detection via keyword matching, profile fact lookup with "chưa ghi nhận" fallback
- **`_estimate_prompt_context_tokens`**: `estimate_tokens(User.md) + estimate_tokens(summary) + sum(estimate_tokens(msg) for msg in recent)`

### `src/benchmark.py`

- **`recall_points`**: Case-insensitive substring matching over `expected_contains` list
- **`heuristic_quality`**: `0.3 + 0.5*(matched_ratio) + length_bonus`, capped at 1.0
- **`format_rows`**: Tabulate if available, markdown fallback otherwise
