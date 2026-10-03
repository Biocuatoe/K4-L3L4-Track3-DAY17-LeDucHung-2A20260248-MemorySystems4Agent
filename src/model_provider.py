"""Model provider abstraction layer.

Supports: openai, custom, gemini, anthropic, ollama, openrouter.
All provider SDKs are lazy-imported inside build_chat_model() so the
offline / no-dependencies path works without any extra packages.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# ProviderConfig
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    """Configuration for a chat model provider.

    Required providers: openai, custom, gemini, anthropic, ollama, openrouter.
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


# ---------------------------------------------------------------------------
# normalize_provider
# ---------------------------------------------------------------------------

_ALIASES: dict[str, str] = {
    # Canonical name aliases
    "google": "gemini",
    "claude": "anthropic",
    "anthorpic": "anthropic",
    "antrhopic": "anthropic",
    "open-ai": "openai",
    "openaiapi": "openai",
    "gpt": "openai",
    "ollama-local": "ollama",
    "local": "ollama",
}

_SUPPORTED = frozenset([
    "openai", "custom", "gemini", "anthropic", "ollama", "openrouter"
])


def normalize_provider(value: str) -> str:
    """Map a provider string to its canonical name.

    Handles aliases (e.g. ``"gpt"`` → ``"openai"``,
    ``"claude"`` → ``"anthropic"``) and validates against known providers.

    Raises
    ------
    ValueError
        If the normalized name is not a supported provider.
    """
    raw = value.strip().lower()
    name = _ALIASES.get(raw, raw)

    if name not in _SUPPORTED:
        raise ValueError(
            f"Unknown provider {value!r} (normalised to {name!r}). "
            f"Supported: {', '.join(sorted(_SUPPORTED))}"
        )
    return name


# ---------------------------------------------------------------------------
# build_chat_model
# ---------------------------------------------------------------------------

_PROVIDER_MODEL_ATTR: dict[str, str] = {
    "openai":     "model",
    "custom":     "model",
    "gemini":     "model",
    "anthropic":  "model",
    "ollama":     "model",
    "openrouter": "model",
}


def build_chat_model(config: ProviderConfig) -> object:
    """Instantiate a LangChain chat model for the selected provider.

    Provider SDKs are imported **lazily** inside this function so that
    importing this module never fails due to a missing package.

    Parameters
    ----------
    config
        Populated :class:`ProviderConfig`.

    Returns
    -------
    object
        A LangChain chat model instance.

    Raises
    ------
    RuntimeError
        If the required SDK package is not installed.
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url or "http://localhost:11434",
        )

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter
        return ChatOpenRouter(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    # Should never reach here because normalize_provider validates the name
    raise RuntimeError(f"Unhandled provider: {provider}")
