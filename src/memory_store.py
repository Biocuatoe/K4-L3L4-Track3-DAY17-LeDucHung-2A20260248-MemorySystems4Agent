"""Memory-store layer for the Day 17 lab.

Provides:
- ``estimate_tokens()`` – heuristic character-based token counter
- ``UserProfileStore`` – read/write/edit persistent ``User.md`` per user
- ``extract_profile_updates()`` – Vietnamese + English fact extractor
- ``summarize_messages()`` – compact older message history
- ``CompactMemoryManager`` – within-thread memory with compaction
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

def estimate_tokens(text: str | None) -> int:
    """Heuristic token estimator.

    Strips whitespace and divides the non-whitespace character count by 4.
    Returns 0 for empty / None input.
    """
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    non_ws = len(re.sub(r"\s+", "", stripped))
    return max(1, round(non_ws / 4))


# ---------------------------------------------------------------------------
# DEFAULT_PROFILE
# ---------------------------------------------------------------------------

DEFAULT_PROFILE = """\
# User Profile

## Identity
- Name: ...

## Current Context
- Location: ...
- Profession: ...

## Preferences
- Response style: ...
- Favorite drink: ...
- Favorite food: ...

## Interests
- Technical interests: ...
- Pets: ...

## Notes
- (no notes yet)
""".strip()


# ---------------------------------------------------------------------------
# UserProfileStore
# ---------------------------------------------------------------------------

class UserProfileStore:
    """Persistent storage for ``User.md`` per user.

    Each user maps to ``root_dir / <safe_user_id> / "User.md"``.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir).resolve()

    @staticmethod
    def _safe_user_id(user_id: str) -> str:
        """Sanitize ``user_id`` to a safe path component."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must not be empty")
        normalized = unicodedata.normalize("NFKC", user_id.strip())
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", normalized)
        if safe.startswith("."):
            raise ValueError(f"user_id {user_id!r} produces dot-prefixed path component")
        if ".." in safe:
            raise ValueError(f"user_id {user_id!r} contains '..' path traversal")
        return safe

    def path_for(self, user_id: str) -> Path:
        """Return the absolute path to the user's ``User.md``."""
        safe = self._safe_user_id(user_id)
        return self.root_dir / safe / "User.md"

    def read_text(self, user_id: str) -> str:
        """Return file content or the empty default profile markdown."""
        path = self.path_for(user_id)
        if not path.is_file():
            return DEFAULT_PROFILE
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown to disk and return the file path."""
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace one occurrence of ``search_text`` in ``User.md``; return True if replaced."""
        path = self.path_for(user_id)
        if not path.is_file():
            return False
        content = path.read_text(encoding="utf-8")
        if search_text not in content:
            return False
        new_content = content.replace(search_text, replacement, 1)
        path.write_text(new_content, encoding="utf-8")
        return True

    def file_size(self, user_id: str) -> int:
        """Return the current file size in bytes, or 0 if missing."""
        path = self.path_for(user_id)
        if not path.is_file():
            return 0
        return path.stat().st_size

    _FIELD_LINE_RE = re.compile(r"^-\s+([^:]+):\s*(.*)$")

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse the current ``User.md`` into a ``{field: value}`` dict."""
        text = self.read_text(user_id)
        if text == DEFAULT_PROFILE:
            return {}
        facts: dict[str, str] = {}
        for line in text.splitlines():
            m = self._FIELD_LINE_RE.match(line.strip())
            if m:
                key = m.group(1).strip().lower()
                value = m.group(2).strip()
                if value and value != "...":
                    facts[key] = value
        return facts

    _FIELD_LABELS: dict[str, str] = {
        "name":                "Name",
        "location":            "Location",
        "profession":          "Profession",
        "response style":      "Response style",
        "favorite drink":      "Favorite drink",
        "favorite food":       "Favorite food",
        "technical interests": "Technical interests",
        "pets":               "Pets",
    }

    def upsert_fact(self, user_id: str, field: str, value: str) -> None:
        """Alias for :meth:`upsert`."""
        self.upsert(user_id, field, value)

    def upsert(self, user_id: str, field: str, value: str) -> None:
        """Set ``field`` to ``value`` in the user's ``User.md``."""
        canonical = field.strip().lower()
        display_label = self._FIELD_LABELS.get(canonical, " ".join(p.capitalize() for p in canonical.split("_")))
        path = self.path_for(user_id)
        if path.is_file():
            content = path.read_text(encoding="utf-8")
        else:
            content = DEFAULT_PROFILE

        field_pattern = re.compile(
            r"^(\s*-\s*" + re.escape(display_label) + r":\s*).+$",
            re.MULTILINE | re.IGNORECASE,
        )
        if field_pattern.search(content):
            # Use a replacement that keeps the label prefix and replaces the value.
            def _repl(m: re.Match) -> str:
                prefix = m.group(1)
                return prefix + value
            new_content, _ = field_pattern.subn(_repl, content)
            if _:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(new_content, encoding="utf-8")
                return

        section_map = {
            "name":                "## Identity",
            "location":            "## Current Context",
            "profession":          "## Current Context",
            "response style":      "## Preferences",
            "favorite drink":      "## Preferences",
            "favorite food":       "## Preferences",
            "technical interests":  "## Interests",
            "pets":               "## Interests",
        }
        target_section = section_map.get(display_label.lower())
        new_line = f"- {display_label}: {value}\n"
        if target_section:
            section_pattern = re.compile(
                r"^(" + re.escape(target_section) + r"\n)((?:\n?-.+\n)*)",
                re.MULTILINE
            )
            m = section_pattern.search(content)
            if m:
                indent_block = m.group(2)
                new_content = (
                    content[:m.end(1)]
                    + indent_block
                    + "  " + new_line.strip() + "\n"
                    + content[m.end():]
                )
            else:
                new_content = content + f"\n\n{target_section}\n  {new_line.strip()}\n"
        else:
            new_content = content + f"\n\n- {display_label}: {value}\n"

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Signal patterns
# ---------------------------------------------------------------------------

# Question pattern — matched against the raw message.
_QUESTION_PATTERNS = re.compile(
    r"(?:"
    r"tên\s+gì|ở\s+đâu|làm\s+gì|ở\s+đâu|"
    r"what(?:'s|\s+is)\s+my\s+name|where\s+am\s+i|"
    r"profession|job|drink|food|interest|career|"
    r"nơi\s+ở|nơi\s+nào|làm\s+gì|nghề\s+gì|sở\s+thích|"
    r"technical\s+interests|sở\s+thích\s+kỹ\s+thuật"
    r")",
    re.IGNORECASE
)

# Assertion verbs — allow extraction to proceed even if question words are present.
_ASSERTION_VERBS = re.compile(
    r"(?:"
    r"là|đang|ở|muốn|thích|có|nói|gửi|tell|know|say|"
    r"am\s+(?:an?\s+Engineer)|"
    r"(?:tên|mình)\s+(?:là|mình)\s+"
    r")",
    re.IGNORECASE
)

# Hypothetical / joking statements — skip entire message.
_HYPOTHETICAL_SIGNALS = re.compile(
    r"(?:"
    r"đùa|đùa\s+(?:với|bảo)|joke|hay\s+là|what\s+if|nếu\s+mà|giả\s+sử|"
    r"chỉ\s+là\s+(?:đùa|trêu)"
    r")",
    re.IGNORECASE
)

# Temporary meeting / event locations — skip location field.
_TEMPORARY_LOCATION_SIGNALS = re.compile(
    r"(?:"
    r"đi\s+họp\s+ở|họp\s+tại|công\s+tác\s+ở|đi\s+đến|"
    r"bay\s+ra|hội\s+thảo\s+tại|event\s+at|meeting\s+in|"
    r"đi\s+(?:họp|sự kiện|workshop|talk|conference)"
    r")",
    re.IGNORECASE
)

# Correction / update signals — if a location appears before this, it is old data.
_CORRECTION_SIGNALS = re.compile(
    r"(?:"
    r"đính\s+chính|thông\s+tin\s+mới|không\s+còn\s+|giờ\s+là\s*|"
    r"chuyển\s+sang|thực\s+ra|actually\s+i|hiện\s+tại\s+mình\s+ở|"
    r"(?:mình|ta|tôi)\s+đang\s+ở|"
    r"(?:nhưng|thế\s+mà)\s+(?:lại|thực\s+tế)\s+(?:là|ở)"
    r")",
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Extraction patterns
# Each pattern captures a structured VALUE (noun phrase, not conversational text).
# All patterns require a specific anchor and use lookahead / post-processing
# to ensure the captured value is a clean noun phrase.
# ---------------------------------------------------------------------------
_EXTRACTION_PATTERNS = [
    # Name — preceded by explicit self-introduction only.
    (re.compile(
        r"(?:^|[\s,.])"
        r"(?:"
        r"tên\s+(?:là|mình\s+là|em\s+là|tôi\s+là)\s*|"
        r"my\s+name\s+(?:is|called)\s*|"
        r"i(?:'m|'m\s+called| am called)\s*|"
        r"called\s+"
        r")"
        r"([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9\s]{0,30})",
        re.IGNORECASE
    ), "name"),

    # Location — strong residence anchors only. Skips deictics (đây/đó) and
    # temporary meeting locations. The correction-aware filter below handles
    # old-location-before-correction cases.
    (re.compile(
        r"(?:^|[\s,.])"
        r"(?:"
        r"(?:hiện\s+tại|bây\s+giờ|hiện\s+tại\s+mình|giờ\s+mình)\s+ở\s*|"
        r"(?:đang\s+ở|hiện\s+ở)\s*|"
        r"(?:làm\s+việc)\s+ở\s*|"
        r"(?:mình\s+)ở\s*"
        r")"
        r"([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9\s]{1,40}?)"
        r"(?=\s*(?:mấy|trong|vài|về|để|còn|và|thì|nhưng|,|\.|$))",
        re.IGNORECASE
    ), "location"),

    # Profession — explicit job introduction only.
    # Must start with a letter (not a Vietnamese noun like "việc").
    (re.compile(
        r"(?:^|[\s,.])"
        r"(?:"
        r"(?:đang\s+)?làm\s+(?:nghề\s+)?|"
        r"(?:đang\s+)?làm\s+việc\s+(?:với\s+)?|"
        r"working\s+as\s+(?:an?\s+)?|"
        r"job\s+is\s+(?:an?\s+)?|"
        r"nghề\s+(?:của\s+)?(?:mình|tôi)\s+là\s*|"
        r"(?:i\s+)?am\s+(?:a\s+)?"
        r")"
        r"([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9\s/+-]{2,50}?)"
        r"(?=\s*(?:cho|với|tại|ở|của|,|\.|$))",
        re.IGNORECASE
    ), "profession"),

    # Response style — the captured value ALWAYS contains a style keyword
    # (bullet / ngắn / concise / short). Anchors are strict preference indicators.
    (re.compile(
        r"(?:"
        # "muốn trả lời X" / "thích trả lời X" / "want reply X"
        r"(?:muốn|thích|want|prefer)\s+(?:trả\s+lời|reply)\s+(?:bằng\s+)?(?:theo\s+)?|"
        # "trả lời X" / "reply X" / "trả lời thành X"
        r"(?:trả\s+lời|reply)\s+(?:bằng\s+)?(?:theo\s+)?(?:thành\s+)?|"
        # "style X" / "kiểu trả lời X"
        r"(?:style|kiểu)\s+(?:trả\s+lời\s+)?|"
        # "theo dạng X" / "as form X"
        r"(?:theo|as)\s+(?:dạng|kiểu|form)\s*"
        r")"
        r"("
        r"(?:3|three|ba)\s+bullet[^\n?,;]{0,40}?|"
        r"bullet[^\n?,;]{0,40}?|"
        r"ngắn\s+gọn[^\n?,;]{0,40}?|"
        r"concise[^\n?,;]{0,40}?|"
        r"short[^\n?,;]{0,40}?"
        r")"
        r"(?=\s*(?:thôi|nhé|nha|hay\s+không|,|\.|;|$))",
        re.IGNORECASE
    ), "response style"),

    # Favorite drink — noun phrase beverage name only.
    (re.compile(
        r"(?:"
        r"(?:đồ\s+)?uống\s+(?:yêu\s+thích\s+)?(?:là\s+)?|"
        r"(?:thích|uống|yêu\s+thích)\s+(?:đồ\s+)?uống\s+(?:là\s+)?|"
        r"(?:đồ\s+uống\s+)?yêu\s+thích\s+(?:là\s+)?|"
        r"favorite\s+drink\s+(?:is\s+|of\s+choice\s+)?"
        r")"
        r"([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9\s]{1,50}?)"
        r"(?=\s*(?:thôi|nhé|nha|,|\.|;|$))",
        re.IGNORECASE
    ), "favorite drink"),

    # Favorite food — noun phrase food name only.
    (re.compile(
        r"(?:"
        r"(?:món\s+)?ăn\s+(?:yêu\s+thích\s+)?(?:là\s+)?|"
        r"(?:thích|yêu\s+thích)\s+(?:món\s+)?(?:uống\s+)?(?:là\s+)?|"
        r"(?:món\s+ăn\s+)?yêu\s+thích\s+(?:là\s+)?|"
        r"favorite\s+food\s+(?:is\s+|of\s+choice\s+)?"
        r")"
        r"([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9\s]{1,50}?)"
        r"(?=\s*(?:thôi|nhé|nha|,|\.|;|$))",
        re.IGNORECASE
    ), "favorite food"),

    # Technical interests — comma-separated interest list. Post-filtered below.
    (re.compile(
        r"(?:"
        r"(?:mối\s+quan\s+tâm|mình\s+quan\s+tâm)\s+(?:về\s+)?|"
        r"(?:quan\s+tâm|tập\s+trung|focus)\s+(?:vào\s+)?(?:là\s+)?|"
        r"(?:interested|intersted)\s+(?:về\s+)?(?:in\s+)?|"
        r"technical\s+interests?\s+(?:là\s+)?|"
        r"(?:dài\s+hạn)\s*:\s*"
        r")"
        r"([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9\s,]{3,100}?)"
        r"(?=\s*(?:nên|thì|và|còn|Tạm|còn\s+là|,|\.|\?|;|$))",
        re.IGNORECASE
    ), "technical interests"),

    # Pets — after explicit pet-ownership keywords.
    (re.compile(
        r"(?:"
        r"(?:mình|em|tôi)\s+nuôi\s+(?:một\s+bé\s+)?|"
        r"(?:có\s+)?(?:một\s+bé|một\s+con)\s+|"
        r"(?:pet|cũng\s+nuôi)\s+(?:một\s+bé\s+)?|"
        r"(?:tên\s+(?:bé|bạn|nó))\s+"
        r")"
        r"([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9\s]{1,40}?)"
        r"(?=\s*(?:mà|bé|nhé|nha|,|\.|;|$))",
        re.IGNORECASE
    ), "pets"),
]


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract stable profile facts from a user message.

    Returns a dict mapping field names to extracted values.
    Only returns facts from assertive statements; skips questions,
    jokes, temporary locations, and conversational fragments.
    """
    if not message or not message.strip():
        return {}

    msg = message.strip()

    # Skip pure questions — also skip if message is a question about a specific field.
    if msg.endswith("?") or _QUESTION_PATTERNS.search(msg):
        if not _ASSERTION_VERBS.search(msg):
            return {}

    # Skip obvious jokes / hypotheticals.
    if _HYPOTHETICAL_SIGNALS.search(msg):
        return {}

    results: dict[str, str] = {}

    for pattern, field in _EXTRACTION_PATTERNS:
        # Use finditer to find ALL matches (handles correction with old+new location).
        for m in pattern.finditer(msg):
            raw = m.group(1) if m.lastindex and m.group(1) is not None else None
            if raw is None:
                continue
            value = raw.strip()
            # Strip trailing punctuation.
            value = re.sub(r"[,.\s]+$", "", value)
            # Strip trailing conjunction phrases.
            value = re.sub(
                r"\s+(?:cho\s+\w+|mà\s+\S+|và\s+\S+|hay\s+\S+|là\s+\S+|"
                r"rồi|nhé|nha|thôi|thì|nên)\s*$",
                "", value, flags=re.IGNORECASE
            )
            value = value.strip()

            if not value or len(value) < 2:
                continue
            if value.endswith("?") or value.startswith("?"):
                continue
            if len(value) > 60:
                continue
            word_count = len(value.split())

            # Reject single-word values for fields that always need multi-word descriptions.
            # NOTE: name, location, profession, pets can legitimately be single words.
            if field in ("technical interests", "response style"):
                if word_count < 2:
                    continue

            # Technical interests: reject conversational fragments.
            # MUST use re.match (not search) to reject values STARTING with pronouns.
            if field == "technical interests":
                stripped_v = value.strip().lower()
                # Reject if starts with a personal pronoun.
                if re.match(r"^(?:mình|tôi|ta|em)\s+", stripped_v):
                    continue
                # Also reject if the pattern matches "pronoun + verb" somewhere in the value.
                if re.search(r"^(?:mình|tôi|ta|em)\s+(?:thích|muốn|là|ở|có)\s+", stripped_v):
                    continue
                # Reject values that are self-referential questions like "kỹ thuật của mình".
                if re.search(r"^(?:của\s+)?(?:mình|tôi|ta|em|chúng\s+ta)", stripped_v):
                    continue

            # Favorite drink / food / technical interests: reject values containing response style keywords.
            if field in ("favorite drink", "favorite food", "technical interests"):
                val_lower = value.strip().lower()
                if re.search(r"\b(?:bullet|ngắn\s+gọn|concise|short)\b", val_lower):
                    continue
                if re.search(r"\b(?:trade-off|ví\s+dụ|thực\s+chiến)\b", val_lower):
                    continue

            # Response style: reject reaffirmation values.
            if field == "response style":
                if re.search(r"vẫn\s+giữ|cũng\s+vẫn", value, re.IGNORECASE):
                    continue
                rs = value.strip().lower()
                # Reject conversational sentence fragments.
                if re.match(r"^(?:vì|bởi|nhưng|cho\s+nên)", rs):
                    continue
                # Reject if contains sentence-structure words (mà, nên, thì, rằng).
                if re.findall(r"\s(mà|nên|thì|rằng)\s", rs):
                    continue
                # Reject if starts with pronoun + verb pattern.
                if re.match(r"^(?:mình|tôi|ta|em|bạn)\s+(?:\w+\s+){0,3}(?:thì|để|nên|mà)", rs):
                    continue

            # Profession: reject single-word and short false captures.
            if field == "profession":
                if word_count < 2:
                    continue
                if len(value) < 6:
                    continue

            # Location: skip temporary meeting locations.
            if field == "location" and _TEMPORARY_LOCATION_SIGNALS.search(msg):
                continue

            # Location: skip deictic pronouns (đây, đó, này).
            if field == "location":
                if re.match(r"^(?:đây|đó|này)\s*$", value.strip(), re.IGNORECASE):
                    continue
                stripped_l = value.strip().lower()
                if len(stripped_l) <= 3 and not re.search(r"[A-Za-zÀ-ỹ]{2,}", stripped_l):
                    continue

            # Location correction-aware filter:
            # If a correction signal (thực ra, không còn, ...) is present,
            # skip any location value that appears BEFORE the correction.
            if field == "location" and _CORRECTION_SIGNALS.search(msg):
                correction_match = _CORRECTION_SIGNALS.search(msg)
                value_lower = value.lower()
                positions = [pm.start() for pm in re.finditer(re.escape(value_lower), msg.lower())]
                if positions and all(p < correction_match.start() for p in positions):
                    continue

            results[field] = value

    return results


# ---------------------------------------------------------------------------
# summarize_messages
# ---------------------------------------------------------------------------

def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary of older messages (max ~600 chars)."""
    if not messages:
        return ""

    seen: set[str] = set()
    items: list[str] = []

    for entry in messages:
        role = entry.get("role", "user")
        content = entry.get("content", "")
        if not content.strip():
            continue

        if role == "user":
            truncated = content.strip()[:80]
            if truncated not in seen:
                seen.add(truncated)
                items.append(truncated)
        else:
            brief = "-> " + content.strip()[:40]
            if brief not in seen:
                seen.add(brief)
                items.append(brief)

    kept = items[:max_items]
    result = "Da thao luan: " + ", ".join(kept)
    if len(result) > 600:
        result = result[:597] + "..."
    return result


# ---------------------------------------------------------------------------
# CompactMemoryManager
# ---------------------------------------------------------------------------

class CompactMemoryManager:
    """Within-thread memory that auto-compacts when token budget is exceeded."""

    def __init__(self, threshold_tokens: int, keep_messages: int) -> None:
        self.threshold_tokens = threshold_tokens
        self.keep_messages = keep_messages
        self._state: dict[str, dict[str, object]] = {}

    def _get_state(self, thread_id: str) -> dict[str, dict[str, object]]:
        if thread_id not in self._state:
            self._state[thread_id] = {"messages": [], "summary": "", "compactions": 0}
        return self._state[thread_id]

    def _total_tokens(self, state: dict[str, object]) -> int:
        total = estimate_tokens(state.get("summary", ""))
        for m in state.get("messages", []):
            total += estimate_tokens(m.get("content", ""))
        return total

    def append(self, thread_id: str, role: str, content: str) -> None:
        state = self._get_state(thread_id)
        messages: list[dict[str, str]] = state["messages"]
        messages.append({"role": role, "content": content})

        if self._total_tokens(state) > self.threshold_tokens:
            to_summarize = messages[:-self.keep_messages]
            if to_summarize:
                old_summary: str = state["summary"]
                new_summary = summarize_messages(to_summarize)
                merged = (old_summary + " | " + new_summary) if old_summary else new_summary
                if len(merged) > 600:
                    merged = merged[:597] + "..."
                state["summary"] = merged
                state["messages"] = messages[-self.keep_messages:]
                state["compactions"] = state.get("compactions", 0) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        state = self._get_state(thread_id)
        return {
            "messages": list(state["messages"]),
            "summary": state.get("summary", ""),
            "compactions": state.get("compactions", 0),
        }

    def compaction_count(self, thread_id: str) -> int:
        return self._get_state(thread_id).get("compactions", 0)
