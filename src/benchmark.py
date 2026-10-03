"""Benchmark suite for the Day 17 memory-systems lab.

Compares BaselineAgent vs AdvancedAgent on:
- Standard benchmark (data/conversations.json)
- Long-context stress benchmark (data/advanced_long_context.json)

Required columns:
  Agent | Agent tokens only | Prompt tokens processed | Cross-session recall |
  Response quality | Memory growth (bytes) | Compactions
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


# ---------------------------------------------------------------------------
# Datatypes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRow:
    """One row in the benchmark results table."""
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


# ---------------------------------------------------------------------------
# load_conversations
# ---------------------------------------------------------------------------

def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Load conversation JSON from disk.

    Each entry must have ``id``, ``user_id``, ``turns`` (list[str]),
    and ``recall_questions`` (list of dicts with ``question`` and
    ``expected_contains``).
    """
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {path}, got {type(data).__name__}")

    for i, conv in enumerate(data):
        if not isinstance(conv, dict):
            raise ValueError(f"Conversation {i} is not a dict")
        for key in ("id", "user_id", "turns", "recall_questions"):
            if key not in conv:
                raise ValueError(f"Conversation {i} missing key {key!r}")

    return data


# ---------------------------------------------------------------------------
# recall_points
# ---------------------------------------------------------------------------

def recall_points(answer: str, expected: list[str]) -> float:
    """Score how many expected substrings appear in the answer.

    Returns a float in [0.0, 1.0] representing the fraction of expected
    substrings found (case-insensitive substring match).
    Returns 1.0 when ``expected`` is empty.
    Returns 0.0 when ``answer`` is empty.
    """
    if not expected:
        return 1.0
    if not answer:
        return 0.0

    answer_lower = answer.lower()
    matched = sum(1 for exp in expected if exp.lower() in answer_lower)
    return matched / len(expected)


# ---------------------------------------------------------------------------
# heuristic_quality
# ---------------------------------------------------------------------------

def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for an offline-mode answer.

    Scoring rationale
    -----------------
    - Empty answer: 0.0
    - Non-empty answer: 0.3 base
    - Bonus for matching expected substrings: up to +0.5
    - Length sanity bonus / penalty: prefers answers around ~100 chars
      (|len/1000 - 0.1| small → +0.2, zeroed if negative)
    - Capped at 1.0.
    """
    if not answer:
        return 0.0

    score = 0.3

    # Bonus for expected substring coverage.
    if expected:
        recall = recall_points(answer, expected)
        score += recall * 0.5

    # Length sanity: penalise very long or very short answers.
    # Prefer ~100-character answers.
    length_ratio = len(answer) / 1000.0
    length_bonus = max(0.0, 0.2 - abs(length_ratio - 0.1))
    score += length_bonus

    return min(1.0, score)


# ---------------------------------------------------------------------------
# run_agent_benchmark
# ---------------------------------------------------------------------------

def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate one agent over a list of conversations.

    Parameters
    ----------
    agent_name
        Human-readable name (e.g. "Baseline" or "Advanced").
    agent
        Either a :class:`BaselineAgent` or :class:`AdvancedAgent` instance.
    conversations
        List of conversation dicts as loaded by :func:`load_conversations`.
    config
        :class:`LabConfig` (used to discover the initial profile size for advanced).

    Returns
    -------
    BenchmarkRow
        Aggregated metrics.
    """
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_recall_score = 0.0
    total_quality_score = 0.0
    total_recall_questions = 0

    # Collect all thread IDs used in this run (needed for compaction count).
    all_thread_ids: list[str] = []

    # Track memory growth (only meaningful for Advanced).
    has_memory_file_size = hasattr(agent, "memory_file_size")
    initial_memory_size = 0

    for conv in conversations:
        user_id = conv["user_id"]
        conv_id = conv["id"]
        turns = conv["turns"]
        recall_questions: list[dict[str, Any]] = conv.get("recall_questions", [])

        # Record initial profile size for advanced agent (before any dialogue).
        if has_memory_file_size and total_agent_tokens == 0:
            initial_memory_size = agent.memory_file_size(user_id)

        thread_id = conv_id  # one thread per conversation for the dialogue part.
        all_thread_ids.append(thread_id)

        # --- Feed all dialogue turns -----------------------------------------
        for turn_text in turns:
            result = agent.reply(user_id, thread_id, turn_text)
            total_agent_tokens += result.get("agent_tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)

        # --- Ask recall questions in a fresh thread ---------------------------
        for i, rq in enumerate(recall_questions):
            recall_thread_id = f"{conv_id}-recall-{i}"
            question_text = rq["question"]
            expected: list[str] = rq.get("expected_contains", [])

            result = agent.reply(user_id, recall_thread_id, question_text)
            answer = result.get("answer", "")

            total_agent_tokens += result.get("agent_tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)

            rp = recall_points(answer, expected)
            hq = heuristic_quality(answer, expected)

            total_recall_score += rp
            total_quality_score += hq
            total_recall_questions += 1

            all_thread_ids.append(recall_thread_id)

    # Final memory size for advanced.
    final_memory_size = 0
    if has_memory_file_size:
        # Use the user_id from the last conversation (conventions.json uses dungct;
        # advanced_long_context uses dungct_stress).
        final_memory_size = agent.memory_file_size(user_id)

    memory_growth_bytes = max(0, final_memory_size - initial_memory_size)

    # Compaction count: sum over all unique thread IDs used.
    total_compactions = sum(agent.compaction_count(tid) for tid in all_thread_ids)

    avg_recall = total_recall_score / total_recall_questions if total_recall_questions else 0.0
    avg_quality = total_quality_score / total_recall_questions if total_recall_questions else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth_bytes,
        compactions=total_compactions,
    )


# ---------------------------------------------------------------------------
# format_rows
# ---------------------------------------------------------------------------

def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a markdown table.

    Tries to use ``tabulate`` if available; falls back to a plain markdown table.
    """
    # Check for tabulate.
    tabulate = None
    try:
        from tabulate import tabulate as _tabulate
        tabulate = _tabulate
    except ImportError:
        pass

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]

    def fmt_row(r: BenchmarkRow) -> list[str]:
        return [
            r.agent_name,
            str(r.agent_tokens_only),
            str(r.prompt_tokens_processed),
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            str(r.memory_growth_bytes),
            str(r.compactions),
        ]

    if tabulate is not None:
        table_data = [fmt_row(r) for r in rows]
        return tabulate(table_data, headers=headers, tablefmt="github")

    # Fallback: plain markdown table.
    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        lines.append("| " + " | ".join(fmt_row(r)) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run both benchmark suites and print comparison tables."""
    config = load_config(Path(__file__).resolve().parent.parent)

    data_dir = config.data_dir
    conversations_path = data_dir / "conversations.json"
    stress_path = data_dir / "advanced_long_context.json"

    # Load datasets.
    conversations = load_conversations(conversations_path)
    stress_conversations = load_conversations(stress_path)

    # Create isolated state directories for clean benchmark runs.
    with tempfile.TemporaryDirectory(prefix="day17-bench-") as tmp_base:
        tmp_root = Path(tmp_base)
        from model_provider import ProviderConfig

        bench_config = LabConfig(
            base_dir=config.base_dir,
            data_dir=data_dir,
            state_dir=tmp_root / "state",
            compact_threshold_tokens=600,   # low enough to trigger compaction on stress test
            compact_keep_messages=4,
            model=ProviderConfig(
                provider="openai",
                model_name="gpt-4o-mini",
                temperature=0.0,
                api_key=None,
            ),
            judge_model=ProviderConfig(
                provider="openai",
                model_name="gpt-4o-mini",
                temperature=0.0,
                api_key=None,
            ),
        )

        # Ensure state dir exists.
        bench_config.state_dir.mkdir(parents=True, exist_ok=True)

        # --- Standard Benchmark -----------------------------------------------
        print("## Standard Benchmark\n")
        standard_rows: list[BenchmarkRow] = []

        for agent_name, agent_cls in [("Baseline", BaselineAgent), ("Advanced", AdvancedAgent)]:
            agent = agent_cls(config=bench_config, force_offline=True)
            row = run_agent_benchmark(agent_name, agent, conversations, bench_config)
            standard_rows.append(row)

        print(format_rows(standard_rows))
        print()

        # --- Long-Context Stress Benchmark ------------------------------------
        print("## Long-Context Stress Benchmark\n")

        stress_rows: list[BenchmarkRow] = []
        for agent_name, agent_cls in [("Baseline", BaselineAgent), ("Advanced", AdvancedAgent)]:
            agent = agent_cls(config=bench_config, force_offline=True)
            row = run_agent_benchmark(agent_name, agent, stress_conversations, bench_config)
            stress_rows.append(row)

        print(format_rows(stress_rows))


if __name__ == "__main__":
    main()
