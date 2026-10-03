"""Shared configuration for the Day 17 memory-systems lab.

Loads environment variables and assembles a :class:`LabConfig`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


# ---------------------------------------------------------------------------
# LabConfig
# ---------------------------------------------------------------------------

class LabConfig:
    """Root configuration object for the lab.

    Attributes
    ----------
    base_dir
        Absolute path to the repository root.
    data_dir
        Absolute path to ``base_dir / "data"``.
    state_dir
        Absolute path to ``base_dir / "state"`` (created automatically).
    compact_threshold_tokens
        Token-count threshold that triggers compact memory.
    compact_keep_messages
        Number of recent messages to keep after compaction.
    model
        :class:`ProviderConfig` for the primary agent.
    judge_model
        :class:`ProviderConfig` for the judge / evaluation model.
    """

    __slots__ = (
        "base_dir", "data_dir", "state_dir",
        "compact_threshold_tokens", "compact_keep_messages",
        "model", "judge_model",
    )

    def __init__(
        self,
        *,
        base_dir: Path,
        data_dir: Path,
        state_dir: Path,
        compact_threshold_tokens: int,
        compact_keep_messages: int,
        model: ProviderConfig,
        judge_model: ProviderConfig,
    ) -> None:
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.state_dir = state_dir
        self.compact_threshold_tokens = compact_threshold_tokens
        self.compact_keep_messages = compact_keep_messages
        self.model = model
        self.judge_model = judge_model


# ---------------------------------------------------------------------------
# _load_dotenv
# ---------------------------------------------------------------------------

def _load_dotenv(root: Path) -> None:
    """Attempt to load a .env file if python-dotenv is available."""
    env_path = root / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        # Silently skip when python-dotenv is not installed.
        pass


# ---------------------------------------------------------------------------
# _read_env_str / _read_env_float / _read_env_int
# ---------------------------------------------------------------------------

def _read_env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _read_env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError("must be positive")
        return value
    except (ValueError, OverflowError):
        return default


# ---------------------------------------------------------------------------
# _build_provider_config
# ---------------------------------------------------------------------------

def _build_provider_config(
    provider_env: str,
    model_env: str,
    temperature_env: str,
    default_provider: str,
    default_model: str,
    default_temperature: float,
) -> ProviderConfig:
    """Build a ProviderConfig from environment variables."""
    provider_raw = _read_env_str(provider_env, default_provider)
    provider = normalize_provider(provider_raw)

    model = _read_env_str(model_env, default_model)
    temperature = _read_env_float(temperature_env, default_temperature)

    # Select credential env vars based on the canonical provider name.
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = None
    elif provider == "custom":
        api_key = os.environ.get("CUSTOM_API_KEY")
        base_url = os.environ.get("CUSTOM_BASE_URL")
    elif provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        base_url = None
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        base_url = None
    elif provider == "ollama":
        api_key = None
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    elif provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        base_url = None
    else:
        # Should be unreachable after normalize_provider.
        api_key = None
        base_url = None

    return ProviderConfig(
        provider=provider,
        model_name=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load (or return a default) lab configuration.

    Parameters
    ----------
    base_dir
        Repository root.  Defaults to the directory containing this file.

    Returns
    -------
    LabConfig
        Fully-populated configuration object.
    """
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # -- .env ---------------------------------------------------------------
    _load_dotenv(root)

    # -- Primary model ------------------------------------------------------
    model = _build_provider_config(
        provider_env="LLM_PROVIDER",
        model_env="LLM_MODEL",
        temperature_env="LLM_TEMPERATURE",
        default_provider="openai",
        default_model="gpt-4o-mini",
        default_temperature=0.0,
    )

    # -- Judge model --------------------------------------------------------
    judge_model = _build_provider_config(
        provider_env="JUDGE_PROVIDER",
        model_env="JUDGE_MODEL",
        temperature_env="JUDGE_TEMPERATURE",
        default_provider="openai",
        default_model="gpt-4o-mini",
        default_temperature=0.0,
    )

    # -- Compact memory settings ---------------------------------------------
    compact_threshold = _read_env_int("COMPACT_THRESHOLD_TOKENS", 800)
    compact_keep = _read_env_int("COMPACT_KEEP_MESSAGES", 6)

    # -- Directories --------------------------------------------------------
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    data_dir = root / "data"

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model,
        judge_model=judge_model,
    )
