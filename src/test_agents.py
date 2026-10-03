"""Pytest test suite for the Day 17 memory-systems lab.

Run with:  pytest src/test_agents.py -v

When pytest is not available, the fixture functions and pytest.raises are
provided as standard-library fallbacks so the test module can still be
imported and its functions called manually with a tmp_path-like Path.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import re

# ---------------------------------------------------------------------------
# pytest compatibility shim (works with or without pytest installed)
# ---------------------------------------------------------------------------

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore[assignment, misc]

# Provide a no-op @pytest.fixture decorator when pytest is absent.
if pytest is None:
    def _dummy_fixture(fn):
        """Pass-through decorator when pytest is not available."""
        return fn
    class _PytestRaises:
        def __init__(self, exc_type):
            self._exc_type = exc_type
        def __enter__(self):
            return None
        def __exit__(self, *args):
            return True  # suppress the exception
    class pytest:
        """Shim when pytest is not installed."""
        @staticmethod
        def fixture(fn):
            return _dummy_fixture(fn)
        @staticmethod
        def raises(exc_type):
            return _PytestRaises(exc_type)


from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import ProviderConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_state_dir(tmp_path: Path) -> Path:
    """Return a state directory inside tmp_path."""
    d = tmp_path / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def bench_config(tmp_state_dir: Path) -> LabConfig:
    """Build an isolated config with low compact threshold."""
    return LabConfig(
        base_dir=tmp_state_dir.parent,
        data_dir=tmp_state_dir.parent / "data",
        state_dir=tmp_state_dir,
        compact_threshold_tokens=120,
        compact_keep_messages=3,
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


# ---------------------------------------------------------------------------
# test_user_markdown_read_write_edit
# ---------------------------------------------------------------------------

def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """User.md can be created, read, updated, and edited."""
    store = UserProfileStore(tmp_path / "profiles")

    user_id = "alice"
    initial = "# User Profile\n\n## Identity\n- Name: Alice\n"
    path = store.write_text(user_id, initial)

    assert path.name == "User.md"
    assert path.is_file()

    # Read it back.
    content = store.read_text(user_id)
    assert "Alice" in content

    # Edit a line.
    changed = store.edit_text(user_id, "Alice", "Bob")
    assert changed is True

    updated = store.read_text(user_id)
    assert "Bob" in updated
    assert "Alice" not in updated

    # File size grows after edit.
    size1 = store.file_size(user_id)
    assert size1 > 0


# ---------------------------------------------------------------------------
# test_compact_trigger
# ---------------------------------------------------------------------------

def test_compact_trigger(tmp_path: Path) -> None:
    """Long threads trigger compaction and keep recent messages."""
    cfg = LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=120,
        compact_keep_messages=3,
        model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
        judge_model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
    )
    agent = AdvancedAgent(config=cfg, force_offline=True)

    user_id = "bob"
    thread_id = "thread-1"

    # Send 12 medium-length messages.
    base = "Day la mot tin nhan dai du de tich luy token. "
    for i in range(12):
        agent.reply(user_id, thread_id, base * 6 + f" [msg {i}]")

    count = agent.compaction_count(thread_id)
    assert count > 0, "Compaction should have triggered"

    ctx = agent.compact_memory.context(thread_id)
    messages = ctx["messages"]
    assert len(messages) <= cfg.compact_keep_messages + 2


# ---------------------------------------------------------------------------
# test_cross_session_recall
# ---------------------------------------------------------------------------

def test_cross_session_recall(tmp_path: Path) -> None:
    """Advanced remembers across sessions; Baseline does not."""
    cfg_base = LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=800,
        compact_keep_messages=6,
        model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
        judge_model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
    )

    # ---- Baseline ----
    baseline = BaselineAgent(config=cfg_base, force_offline=True)

    # Thread A: introduce self.
    baseline.reply("alice", "baseline-A", "Mình tên là TestUser.")
    # Thread B: recall name.
    result = baseline.reply("alice", "baseline-B", "Mình tên gì?")
    answer_b = result["answer"]
    assert "TestUser" not in answer_b, (
        "Baseline should NOT recall across threads; got: " + answer_b
    )

    # ---- Advanced ----
    advanced = AdvancedAgent(config=cfg_base, force_offline=True)

    # Thread A: introduce self.
    advanced.reply("alice", "advanced-A", "Mình tên là TestUser.")
    # Thread B: recall name.
    result = advanced.reply("alice", "advanced-B", "Mình tên gì?")
    answer_a = result["answer"]
    assert "TestUser" in answer_a, (
        "Advanced SHOULD recall across threads; got: " + answer_a
    )


# ---------------------------------------------------------------------------
# test_compact_reduces_prompt_load_on_long_thread
# ---------------------------------------------------------------------------

def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Advanced prompt load stays small after compaction; raw total is larger."""
    cfg = LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=150,
        compact_keep_messages=3,
        model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
        judge_model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
    )

    user_id = "charlie"
    thread_id = "long-thread"

    # Track raw tokens.
    raw_tokens = 0
    filler = "Day la mot doan van dai de tang token count. " * 5

    agent = AdvancedAgent(config=cfg, force_offline=True)

    for i in range(20):
        msg = f"{filler} [turn {i}]"
        raw_tokens += estimate_tokens(msg)
        agent.reply(user_id, thread_id, msg)

    # The compacted prompt should be smaller than raw_tokens.
    prompt_tokens = agent.prompt_token_usage(thread_id)
    assert prompt_tokens < raw_tokens, (
        f"Compacted prompt ({prompt_tokens}) should be smaller than "
        f"raw total ({raw_tokens})"
    )
    assert agent.compaction_count(thread_id) >= 2


# ---------------------------------------------------------------------------
# test_empty_message
# ---------------------------------------------------------------------------

def test_empty_message(tmp_path: Path) -> None:
    """Empty / whitespace messages do not crash."""
    cfg = LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=800,
        compact_keep_messages=6,
        model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
        judge_model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
    )

    advanced = AdvancedAgent(config=cfg, force_offline=True)

    # Should not raise.
    r1 = advanced.reply("alice", "t1", "")
    assert "answer" in r1

    r2 = advanced.reply("alice", "t2", "   \n\t  ")
    assert "answer" in r2


# ---------------------------------------------------------------------------
# test_missing_profile_file
# ---------------------------------------------------------------------------

def test_missing_profile_file(tmp_path: Path) -> None:
    """read_text returns the default empty profile for unknown users."""
    store = UserProfileStore(tmp_path / "profiles")
    content = store.read_text("unknown_user_xyz")
    assert "User Profile" in content
    assert store.file_size("unknown_user_xyz") == 0


# ---------------------------------------------------------------------------
# test_missing_field_in_recall
# ---------------------------------------------------------------------------

def test_missing_field_in_recall(tmp_path: Path) -> None:
    """Asking about an unknown field does not crash."""
    cfg = LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=800,
        compact_keep_messages=6,
        model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
        judge_model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
    )
    agent = AdvancedAgent(config=cfg, force_offline=True)
    result = agent.reply("alice", "thread-1", "Mình tên là Alice, ở Hà Nội.")
    # Ask about unknown field.
    result2 = agent.reply("alice", "thread-2", "Sở thích kỹ thuật của mình là gì?")
    assert "chưa" in result2["answer"].lower() or "trong hồ sơ" in result2["answer"].lower()


# ---------------------------------------------------------------------------
# test_repeated_fact_no_duplication
# ---------------------------------------------------------------------------

def test_repeated_fact_no_duplication(tmp_path: Path) -> None:
    """Writing the same fact twice leaves exactly one Name field."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "dave"

    store.upsert(user_id, "name", "Alice")
    store.upsert(user_id, "name", "Alice")

    content = store.read_text(user_id)
    # Count occurrences of "- Name:" lines.
    matches = re.findall(r"- Name:", content)
    assert len(matches) == 1, f"Expected 1 Name field, got {len(matches)} in:\n{content}"


# ---------------------------------------------------------------------------
# test_correction_replaces_old_value
# ---------------------------------------------------------------------------

def test_correction_replaces_old_value(tmp_path: Path) -> None:
    """Correction updates the existing field, does not add a duplicate."""
    cfg = LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=800,
        compact_keep_messages=6,
        model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
        judge_model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
    )
    agent = AdvancedAgent(config=cfg, force_offline=True)

    user_id = "eve"
    # First statement.
    agent.reply(user_id, "t1", "Mình tên là Eve, đang ở Huế.")
    # Correction.
    agent.reply(user_id, "t2", "Mình đính chính: giờ mình ở Đà Nẵng rồi.")

    facts = agent.profile_store.facts(user_id)
    loc = facts.get("location", "")
    assert "Đà Nẵng" in loc, f"Expected Đà Nẵng as location, got: {loc}"


# ---------------------------------------------------------------------------
# test_question_only_no_extraction
# ---------------------------------------------------------------------------

def test_question_only_no_extraction(tmp_path: Path) -> None:
    """A question-only message does not produce extracted facts."""
    del tmp_path  # unused
    updates = extract_profile_updates("Mình tên gì?")
    assert updates == {}, f"Expected no extraction from question, got: {updates}"

    updates2 = extract_profile_updates("Bạn ở đâu?")
    assert updates2 == {}, f"Expected no extraction from question, got: {updates2}"


# ---------------------------------------------------------------------------
# test_temporary_meeting_location_ignored
# ---------------------------------------------------------------------------

def test_temporary_meeting_location_ignored(tmp_path: Path) -> None:
    """Temporary meeting locations are not stored as the user's location."""
    cfg = LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=800,
        compact_keep_messages=6,
        model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
        judge_model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
    )
    agent = AdvancedAgent(config=cfg, force_offline=True)

    user_id = "frank"
    agent.reply(user_id, "t1", "Hôm nay mình đi họp ở Hà Nội.")
    agent.reply(user_id, "t2", "Chiều nay mình công tác ở Sài Gòn.")

    facts = agent.profile_store.facts(user_id)
    loc = facts.get("location", "")
    # Temporary meeting locations should not be stored.
    assert "Hà Nội" not in loc, f"Hà Nội (meeting) should not be stored; got: {loc}"
    assert "Sài Gòn" not in loc, f"Sài Gòn (meeting) should not be stored; got: {loc}"


# ---------------------------------------------------------------------------
# test_joke_profession_not_persisted
# ---------------------------------------------------------------------------

def test_joke_profession_not_persisted(tmp_path: Path) -> None:
    """Joking / hypothetical profession changes are not stored."""
    cfg = LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=800,
        compact_keep_messages=6,
        model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
        judge_model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
    )
    agent = AdvancedAgent(config=cfg, force_offline=True)

    user_id = "greg"
    agent.reply(user_id, "t1", "Mình đùa chuyển sang product manager thôi.")
    agent.reply(user_id, "t2", "Hay là mình làm designer cho vui.")

    facts = agent.profile_store.facts(user_id)
    prof = facts.get("profession", "")
    assert "product manager" not in prof.lower()
    assert "designer" not in prof.lower()


# ---------------------------------------------------------------------------
# test_path_traversal_blocked
# ---------------------------------------------------------------------------

def test_path_traversal_blocked(tmp_path: Path) -> None:
    """Path traversal attempts raise ValueError."""
    del tmp_path  # unused
    store = UserProfileStore(Path(tempfile.gettempdir()) / "profiles")

    with pytest.raises(ValueError):
        store._safe_user_id("../../etc/passwd")

    with pytest.raises(ValueError):
        store._safe_user_id("../../../root/.bashrc")

    with pytest.raises(ValueError):
        store._safe_user_id(".hidden")


# ---------------------------------------------------------------------------
# test_unicode_vietnamese_text
# ---------------------------------------------------------------------------

def test_unicode_vietnamese_text(tmp_path: Path) -> None:
    """Profile handles Vietnamese characters correctly (UTF-8 round-trip)."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "huy"

    content = (
        "# User Profile\n\n"
        "## Identity\n- Name: Hue\n"
        "## Current Context\n- Location: Da Nang\n"
        "## Preferences\n- Favorite drink: ca phe sua da\n"
    )
    store.write_text(user_id, content)

    read_back = store.read_text(user_id)
    assert "Hue" in read_back
    assert "Da Nang" in read_back
    assert "ca phe sua da" in read_back


# ---------------------------------------------------------------------------
# test_compact_multiple_times
# ---------------------------------------------------------------------------

def test_compact_multiple_times(tmp_path: Path) -> None:
    """After enough messages, compaction happens at least twice."""
    cfg = LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=80,
        compact_keep_messages=2,
        model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
        judge_model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
    )
    agent = AdvancedAgent(config=cfg, force_offline=True)

    user_id = "ivan"
    thread_id = "many-msgs"
    filler = "A" * 100

    for i in range(30):
        agent.reply(user_id, thread_id, f"{filler} turn {i}")

    count = agent.compaction_count(thread_id)
    assert count >= 2, f"Expected >= 2 compactions, got {count}"


# ---------------------------------------------------------------------------
# test_very_long_single_message
# ---------------------------------------------------------------------------

def test_very_long_single_message(tmp_path: Path) -> None:
    """A single very long message (10 KB) does not crash."""
    cfg = LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=800,
        compact_keep_messages=6,
        model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
        judge_model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
    )
    agent = AdvancedAgent(config=cfg, force_offline=True)

    long_msg = "x" * 10_000
    result = agent.reply("alice", "t1", long_msg)
    assert "answer" in result


# ---------------------------------------------------------------------------
# test_zero_message_context
# ---------------------------------------------------------------------------

def test_zero_message_context(tmp_path: Path) -> None:
    """Brand-new thread has empty messages and zero compactions."""
    cfg = LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=800,
        compact_keep_messages=6,
        model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
        judge_model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
    )
    agent = AdvancedAgent(config=cfg, force_offline=True)

    ctx = agent.compact_memory.context("brand-new-thread")
    assert ctx["messages"] == []
    assert ctx["compactions"] == 0
    assert agent.compaction_count("brand-new-thread") == 0


# ---------------------------------------------------------------------------
# test_baseline_no_persistent_file
# ---------------------------------------------------------------------------

def test_baseline_no_persistent_file(tmp_path: Path) -> None:
    """Baseline agent does not create any User.md files."""
    cfg = LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=800,
        compact_keep_messages=6,
        model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
        judge_model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
    )
    baseline = BaselineAgent(config=cfg, force_offline=True)

    baseline.reply("alice", "t1", "Mình tên là Bob.")
    baseline.reply("alice", "t2", "Mình ở Hà Nội.")

    # No profiles directory should exist.
    profiles_dir = tmp_path / "state" / "profiles"
    if profiles_dir.exists():
        md_files = list(profiles_dir.rglob("User.md"))
        assert len(md_files) == 0, f"Baseline should not create User.md; found: {md_files}"
