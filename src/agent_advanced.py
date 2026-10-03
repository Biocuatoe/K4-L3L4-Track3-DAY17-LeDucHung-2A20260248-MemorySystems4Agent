"""Advanced Agent (Agent B) — short-term + persistent User.md + compact memory.

Features:
- Within-thread compact memory (CompactMemoryManager)
- Persistent profile via User.md (UserProfileStore)
- Cross-session recall through User.md
- Correction handling (last-known-valid-value per field)
- Confidence filtering (skips question-only, joke, hypothetical turns)
"""

from __future__ import annotations

from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)


# ---------------------------------------------------------------------------
# AdvancedAgent
# ---------------------------------------------------------------------------

class AdvancedAgent:
    """Advanced agent — three memory layers.

    1. Within-thread compact memory (:class:`CompactMemoryManager`)
    2. Persistent ``User.md`` (:class:`UserProfileStore`)
    3. Short-term session state (thread → user mapping, token accumulators)

    This agent remembers stable facts across different ``thread_id`` values
    via the persistent ``User.md`` file.
    """

    def __init__(
        self,
        config: LabConfig | None = None,
        force_offline: bool = False,
    ) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline

        # Persistent profile store: state_dir / "profiles" / <user_id> / "User.md"
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")

        # Compact memory manager.
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )

        # Per-thread token accumulators.
        self.thread_tokens: dict[str, int] = {}       # agent-generated tokens per thread
        self.thread_prompt_tokens: dict[str, int] = {}  # prompt context tokens per thread

        # Map thread_id → user_id (needed for prompt_token_usage lookups).
        self._thread_user: dict[str, str] = {}

        # Optional LangChain agent handle.
        self.langchain_agent: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reply(
        self,
        user_id: str,
        thread_id: str,
        message: str,
    ) -> dict[str, Any]:
        """Route to live or offline path and return a result dict.

        Returns
        -------
        dict
            Keys: ``answer`` (str), ``thread_id`` (str),
            ``agent_tokens`` (int), ``prompt_tokens`` (int),
            ``memory_path`` (str).
        """
        # Record thread → user mapping.
        self._thread_user[thread_id] = user_id

        if self.langchain_agent is not None and not self.force_offline:
            answer = self._live_reply(user_id, thread_id, message)
            return {
                "answer": answer,
                "thread_id": thread_id,
                "agent_tokens": estimate_tokens(answer),
                "prompt_tokens": self.prompt_token_usage(thread_id),
                "memory_path": str(self.profile_store.path_for(user_id)),
            }
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Cumulative estimated token count for assistant replies in this thread."""
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        """Estimated prompt-context tokens for this thread.

        Includes User.md + compact summary + recent messages.
        """
        user_id = self._thread_user.get(thread_id)
        if user_id is None:
            return 0
        return self._estimate_prompt_context_tokens(user_id, thread_id)

    def memory_file_size(self, user_id: str) -> int:
        """Size in bytes of the user's ``User.md``, or 0 if missing."""
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        """Number of compactions performed for this thread."""
        return self.compact_memory.compaction_count(thread_id)

    # ------------------------------------------------------------------
    # Offline path
    # ------------------------------------------------------------------

    def _reply_offline(
        self,
        user_id: str,
        thread_id: str,
        message: str,
    ) -> dict[str, Any]:
        """Advanced offline reply: extract → persist → compact → respond."""
        # 1. Extract stable facts and upsert into User.md.
        updates = extract_profile_updates(message)
        for field, value in updates.items():
            self.profile_store.upsert_fact(user_id, field, value)

        # 2. Append user message to compact memory.
        self.compact_memory.append(thread_id, "user", message)

        # 3. Estimate prompt context tokens.
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        # 4. Generate deterministic response.
        answer = self._offline_response(user_id, thread_id, message)
        answer_tokens = estimate_tokens(answer)

        # 5. Append assistant reply to compact memory.
        self.compact_memory.append(thread_id, "assistant", answer)

        # 6. Update token accumulators.
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + answer_tokens
        )
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        return {
            "answer": answer,
            "thread_id": thread_id,
            "agent_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_path": str(self.profile_store.path_for(user_id)),
        }

    def _estimate_prompt_context_tokens(
        self,
        user_id: str,
        thread_id: str,
    ) -> int:
        """Sum of estimated tokens for User.md + compact summary + recent messages."""
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary: str = ctx.get("summary", "")
        messages: list[dict[str, str]] = ctx.get("messages", [])
        total = estimate_tokens(profile_text)
        total += estimate_tokens(summary)
        for m in messages:
            total += estimate_tokens(m.get("content", ""))
        return total

    def _offline_response(
        self,
        user_id: str,
        thread_id: str,
        message: str,
    ) -> str:
        """Deterministic response using persisted ``User.md`` facts.

        Detects the question intent (Vietnamese + English) and answers
        with the appropriate stored fact.  Falls back to a generic
        acknowledgement that incorporates known facts.
        """
        msg_lower = message.lower().strip()
        facts = self.profile_store.facts(user_id)

        # Helper to retrieve a fact with a default.
        def get(key: str) -> str:
            return facts.get(key, "")

        # ---- Detect intent ----------------------------------------------------
        # Name question.
        if any(kw in msg_lower for kw in ["tên gì", "tên mình", "tên tôi", "my name", "what's my name", "what is my name"]):
            name = get("name")
            if name:
                return f"Tên bạn là {name}."
            return "Mình chưa ghi nhận tên trong hồ sơ."

        # Profession / job question.
        if any(kw in msg_lower for kw in ["nghề gì", "làm gì", "làm nghề", "profession", "job is", "what do i do"]):
            prof = get("profession")
            if prof:
                return f"Bạn đang làm nghề {prof}."
            return "Mình chưa ghi nhận nghề nghiệp trong hồ sơ."

        # Location / nơi ở question.
        if any(kw in msg_lower for kw in ["ở đâu", "nơi ở", "ở đâu", "location", "where am i", "ở đâu"]):
            loc = get("location")
            if loc:
                return f"Bạn hiện đang ở {loc}."
            return "Mình chưa ghi nhận nơi ở trong hồ sơ."

        # Drink question.
        if any(kw in msg_lower for kw in ["đồ uống", "uống gì", "drink", "favorite drink"]):
            drink = get("favorite drink")
            if drink:
                return f"Đồ uống yêu thích của bạn là {drink}."
            return "Mình chưa ghi nhận đồ uống yêu thích trong hồ sơ."

        # Food question.
        if any(kw in msg_lower for kw in ["món ăn", "ăn gì", "food", "favorite food"]):
            food = get("favorite food")
            if food:
                return f"Món ăn yêu thích của bạn là {food}."
            return "Mình chưa ghi nhận món ăn yêu thích trong hồ sơ."

        # Response style question.
        if any(kw in msg_lower for kw in ["style", "trả lời", "bullet", "trả lời như thế nào", "response style", "preferred style"]):
            style = get("response style")
            if style:
                return f"Bạn thích style trả lời: {style}."
            return "Mình chưa ghi nhận style trả lời trong hồ sơ."

        # Technical interests question.
        if any(kw in msg_lower for kw in ["kỹ thuật", "quan tâm", "interest", "technical", "mối quan tâm"]):
            interests = get("technical interests")
            if interests:
                return f"Các mối quan tâm kỹ thuật của bạn: {interests}."
            return "Mình chưa ghi nhận mối quan tâm kỹ thuật trong hồ sơ."

        # Pet question.
        if any(kw in msg_lower for kw in ["nuôi", "pet", "corgi", "bé", "thú cưng"]):
            pets = get("pets")
            if pets:
                return f"Bạn nuôi: {pets}."
            return "Mình chưa ghi nhận thông tin về thú cưng trong hồ sơ."

        # Summary / recap question.
        if any(kw in msg_lower for kw in ["tóm tắt", "summary", "recap", "nhắc lại", "nhắc"]):
            name    = get("name")
            loc     = get("location")
            prof    = get("profession")
            style   = get("response style")
            drink   = get("favorite drink")
            food    = get("favorite food")
            parts = []
            if name:   parts.append(f"tên {name}")
            if loc:    parts.append(f"ở {loc}")
            if prof:   parts.append(f"nghề {prof}")
            if style:  parts.append(f"style {style}")
            if drink:  parts.append(f"uống {drink}")
            if food:   parts.append(f"ăn {food}")
            if parts:
                return "Hồ sơ của bạn: " + ", ".join(parts) + "."
            return "Hồ sơ của bạn hiện trống."

        # ---- No specific question — generic acknowledgement with known facts ----
        name = get("name")
        prof = get("profession")
        loc  = get("location")
        style = get("response style")

        if name or prof or loc or style:
            ctx_parts = []
            if name:  ctx_parts.append(f"tên {name}")
            if loc:   ctx_parts.append(f"ở {loc}")
            if prof:  ctx_parts.append(f"nghề {prof}")
            return f"Ghi nhận. Mình sẽ dùng hồ sơ của bạn ({', '.join(ctx_parts)}) cho các lượt sau."

        return "Đã ghi nhận."

    # ------------------------------------------------------------------
    # Optional LangChain / LangGraph path
    # ------------------------------------------------------------------

    def _maybe_build_langchain_agent(self) -> None:
        """Attempt to build a LangChain/LangGraph agent.

        On any ImportError or exception, silently sets ``self.langchain_agent = None``.
        """
        try:
            from model_provider import build_chat_model
            model = build_chat_model(self.config.model)
            self.langchain_agent = model
        except Exception:
            self.langchain_agent = None

    def _live_reply(
        self,
        user_id: str,
        thread_id: str,
        message: str,
    ) -> str:
        """Invoke the live LangChain model (only when ``langchain_agent`` is set)."""
        if self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)["answer"]
        try:
            from langchain_core.messages import HumanMessage
            response = self.langchain_agent.invoke([HumanMessage(content=message)])
            return str(response.content if hasattr(response, "content") else response)
        except Exception:
            return self._reply_offline(user_id, thread_id, message)["answer"]
