"""Baseline Agent (Agent A) — within-session memory only.

Does NOT persist ``User.md``; forgets long-term facts when thread changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens


# ---------------------------------------------------------------------------
# SessionState
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """Holds all per-thread state for the baseline agent."""
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


# ---------------------------------------------------------------------------
# BaselineAgent
# ---------------------------------------------------------------------------

class BaselineAgent:
    """Baseline agent — short-term memory only.

    Facts do NOT persist across different ``thread_id`` values.
    No ``User.md`` is created.
    """

    def __init__(
        self,
        config: LabConfig | None = None,
        force_offline: bool = False,
    ) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route to live or offline path and return a result dict.

        Returns
        -------
        dict
            Keys: ``answer`` (str), ``thread_id`` (str),
            ``agent_tokens`` (int), ``prompt_tokens`` (int).
        """
        if self.langchain_agent is not None and not self.force_offline:
            # Live path (LangChain / LangGraph) — only used when SDKs present.
            answer = self._live_reply(user_id, thread_id, message)
            return {
                "answer": answer,
                "thread_id": thread_id,
                "agent_tokens": estimate_tokens(answer),
                "prompt_tokens": self.prompt_token_usage(thread_id),
            }
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Cumulative estimated token count for assistant replies in this thread."""
        session = self.sessions.get(thread_id)
        if session is None:
            return 0
        return session.token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        """Cumulative estimated token count for the full prompt context in this thread."""
        session = self.sessions.get(thread_id)
        if session is None:
            return 0
        return session.prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        """Baseline never compacts."""
        return 0

    # ------------------------------------------------------------------
    # Offline path
    # ------------------------------------------------------------------

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Generate a deterministic short reply using only within-thread history."""
        # Ensure session exists.
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()

        session = self.sessions[thread_id]

        # Append user message.
        session.messages.append({"role": "user", "content": message})
        session.prompt_tokens_processed += estimate_tokens(message)

        # Generate reply.
        answer = self._generate_reply(thread_id, message)
        answer_tokens = estimate_tokens(answer)

        # Append assistant message.
        session.messages.append({"role": "assistant", "content": answer})
        session.prompt_tokens_processed += answer_tokens
        session.token_usage += answer_tokens

        return {
            "answer": answer,
            "thread_id": thread_id,
            "agent_tokens": answer_tokens,
            "prompt_tokens": session.prompt_tokens_processed,
        }

    def _generate_reply(self, thread_id: str, message: str) -> str:
        """Deterministic reply based on the conversation history in this thread.

        Looks for self-introductions, location statements, etc. within the
        current thread only.  Returns a short acknowledgement when no match.
        """
        msg_lower = message.lower().strip()

        # Short-circuit: look for "tên" / "name" questions.
        if any(kw in msg_lower for kw in ["tên gì", "tên mình là gì", "tên tôi là gì", "my name is", "what's my name"]):
            for entry in self.sessions[thread_id].messages:
                if entry["role"] != "user":
                    continue
                content = entry["content"]
                # Match self-introduction patterns.
                intro = self._find_name_in_text(content)
                if intro:
                    return f"Tên bạn là {intro}."
            return "Mình chưa biết tên bạn trong phiên này."

        # Location question.
        if any(kw in msg_lower for kw in ["ở đâu", "ở đâu", "nơi ở", "ở đâu", "location", "where am i"]):
            for entry in self.sessions[thread_id].messages:
                if entry["role"] != "user":
                    continue
                loc = self._find_location_in_text(entry["content"])
                if loc:
                    return f"Bạn đang ở {loc}."
            return "Mình chưa biết bạn đang ở đâu trong phiên này."

        # Generic acknowledgement with first 60 chars.
        truncated = message.strip()[:60]
        return f"Đã ghi nhận: {truncated}"

    # ------------------------------------------------------------------
    # Text-search helpers (thread-scoped only)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_name_in_text(text: str) -> str | None:
        """Extract a name from self-introduction text."""
        import re
        patterns = [
            r"(?:tên|mình|tôi)\s+(?:là|mình\s+là)\s+([A-Za-zÀ-ÿ\u00C0-\u024F\u1EA0-\u1EF9]+(?:\s+[A-Za-zÀ-ÿ\u00C0-\u024F\u1EA0-\u1EF9]+)*)",
            r"(?:my\s+name\s+is|i(?:'m|'m\s+called| am called)\s+)([A-Za-z]+(?:\s+[A-Za-z]+)*)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    @staticmethod
    def _find_location_in_text(text: str) -> str | None:
        """Extract a location from within-thread text."""
        import re
        # Only match current/stable location indicators.
        patterns = [
            r"(?:ở|đang\s+ở|hiện\s+tại\s+ở)\s+([A-Za-zÀ-ÿ\s]+?)(?:\s*,|\s*\n|\s*(?:mà|và|nhưng|chứ|\.|$))",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    # ------------------------------------------------------------------
    # Optional LangChain / LangGraph path
    # ------------------------------------------------------------------

    def _maybe_build_langchain_agent(self) -> None:
        """Attempt to build a LangChain/LangGraph agent.

        On any ImportError or exception, silently sets ``self.langchain_agent = None``.
        """
        try:
            from langchain_core.messages import HumanMessage
            from model_provider import build_chat_model
            model = build_chat_model(self.config.model)
            self.langchain_agent = model
        except Exception:
            self.langchain_agent = None

    def _live_reply(self, user_id: str, thread_id: str, message: str) -> str:
        """Invoke the live LangChain model (only when ``langchain_agent`` is set)."""
        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)["answer"]
        try:
            from langchain_core.messages import HumanMessage
            response = self.langchain_agent.invoke([HumanMessage(content=message)])
            return str(response.content if hasattr(response, "content") else response)
        except Exception:
            return self._reply_offline(thread_id, message)["answer"]
