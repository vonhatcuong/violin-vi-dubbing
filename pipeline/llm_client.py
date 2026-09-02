"""Factory for translation + transcription clients — supports Together AI, OpenAI,
Ollama Cloud (OpenAI-compatible), and local faster-whisper."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)


# Ollama: local daemon by default; Ollama Cloud only when base_url points there.
_OLLAMA_LOCAL_BASE_URL = "http://localhost:11434/v1"
_OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"


def _parse_translation_config(cfg: dict[str, Any]) -> tuple[str, str]:
    """Return (provider, model) from the models.translation config entry.

    Supports both the new dict format and the legacy plain-string format:
        # new
        translation:
          provider: openai
          model: gpt-4.1
        # legacy (treated as together)
        translation: "Qwen/Qwen3.5-397B-A17B"
    """
    entry = cfg["models"]["translation"]
    if isinstance(entry, dict):
        return entry.get("provider", "together"), entry["model"]
    return "together", entry


def get_translation_model(cfg: dict[str, Any]) -> str:
    """Return the model name string for translation."""
    _, model = _parse_translation_config(cfg)
    return model


def get_translation_provider(cfg: dict[str, Any]) -> str:
    """Return 'openai', 'together', 'ollama' or 'openai_compat'."""
    provider, _ = _parse_translation_config(cfg)
    return provider


def _entry_dict(cfg: dict[str, Any], section: str) -> dict[str, Any]:
    entry = cfg["models"].get(section)
    return entry if isinstance(entry, dict) else {}


def _make_openai_compat_client(
    provider: str,
    *,
    entry: dict[str, Any] | None = None,
    openai_key_override: str | None = None,
    ollama_key_override: str | None = None,
):
    """Build an OpenAI SDK client for `openai`, `ollama` or `openai_compat`.

    ollama        → base_url = models.*.base_url | $OLLAMA_BASE_URL | localhost.
                    A key is only required when base_url is Ollama Cloud.
    openai_compat → any OpenAI-compatible server (vLLM, LiteLLM, llama.cpp);
                    base_url = models.*.base_url | $OPENAI_COMPAT_BASE_URL (required).
    """
    from openai import OpenAI

    entry = entry or {}
    if provider == "ollama":
        base_url = (entry.get("base_url") or os.environ.get("OLLAMA_BASE_URL") or _OLLAMA_LOCAL_BASE_URL)
        api_key = ollama_key_override or entry.get("api_key") or os.environ.get("OLLAMA_API_KEY")
        if base_url.rstrip("/") == _OLLAMA_CLOUD_BASE_URL and not api_key:
            raise RuntimeError(
                "OLLAMA_API_KEY is not set but base_url is Ollama Cloud. "
                "Get one at https://ollama.com/settings/keys or point base_url at a local daemon."
            )
        return OpenAI(api_key=api_key or "ollama", base_url=base_url)

    if provider == "openai_compat":
        base_url = entry.get("base_url") or os.environ.get("OPENAI_COMPAT_BASE_URL")
        if not base_url:
            raise RuntimeError(
                "provider openai_compat needs models.<stage>.base_url or OPENAI_COMPAT_BASE_URL."
            )
        api_key = entry.get("api_key") or os.environ.get("OPENAI_COMPAT_API_KEY") or "none"
        return OpenAI(api_key=api_key, base_url=base_url)

    api_key = openai_key_override or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key)


_OPENAI_COMPAT_PROVIDERS = ("openai", "ollama", "openai_compat")


def make_translation_client(
    cfg: dict[str, Any],
    *,
    together_key_override: str | None = None,
    openai_key_override: str | None = None,
    ollama_key_override: str | None = None,
):
    """Create the appropriate chat client based on the translation provider config."""
    provider, _ = _parse_translation_config(cfg)

    if provider in _OPENAI_COMPAT_PROVIDERS:
        return _make_openai_compat_client(
            provider,
            entry=_entry_dict(cfg, "translation"),
            openai_key_override=openai_key_override,
            ollama_key_override=ollama_key_override,
        )

    from together import Together
    api_key = together_key_override or os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY environment variable is not set.")
    return Together(api_key=api_key)


# ── Chat (video Q&A) client ─────────────────────────────────

def _parse_chat_config(cfg: dict[str, Any]) -> tuple[str, str]:
    """Return (provider, model) from the models.chat config entry."""
    entry = cfg["models"]["chat"]
    if isinstance(entry, dict):
        return entry.get("provider", "together"), entry["model"]
    return "together", entry


def get_chat_provider(cfg: dict[str, Any]) -> str:
    provider, _ = _parse_chat_config(cfg)
    return provider


def get_chat_model(cfg: dict[str, Any]) -> str:
    _, model = _parse_chat_config(cfg)
    return model


def make_chat_client(
    cfg: dict[str, Any],
    *,
    together_key_override: str | None = None,
    openai_key_override: str | None = None,
    ollama_key_override: str | None = None,
):
    """Create the chat client based on ``models.chat.provider``.

    Independent from translation because chat needs a vision-language model;
    the translation provider may not host one.
    """
    provider, _ = _parse_chat_config(cfg)

    if provider in _OPENAI_COMPAT_PROVIDERS:
        return _make_openai_compat_client(
            provider,
            entry=_entry_dict(cfg, "chat"),
            openai_key_override=openai_key_override,
            ollama_key_override=ollama_key_override,
        )

    from together import Together
    api_key = together_key_override or os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY environment variable is not set.")
    return Together(api_key=api_key)


# ── Startup validation ──────────────────────────────────────

_PROVIDER_ENV_KEY = {
    "together":       "TOGETHER_API_KEY",
    "openai":         "OPENAI_API_KEY",
    "elevenlabs":     "ELEVENLABS_API_KEY",
    # ollama: key only needed for Ollama Cloud — checked at client creation
    "ollama":         None,
    "openai_compat":  None,
    # local-only providers — no env key required
    "faster-whisper": None,
    "supertonic":     None,
    "f5vi":           None,
    "vieneu":         None,
}


def required_env_keys(cfg: dict[str, Any]) -> set[str]:
    """Return the env var names required by the active provider config."""
    keys: set[str] = set()

    def _add(provider: str) -> None:
        env = _PROVIDER_ENV_KEY.get(provider)
        if env:
            keys.add(env)

    _add(get_transcription_provider(cfg))
    _add(get_translation_provider(cfg))
    _add(get_chat_provider(cfg))

    tts_entry = cfg["models"].get("tts")
    if isinstance(tts_entry, dict):
        tts_provider = tts_entry.get("provider", "together")
    else:
        tts_provider = "together"
    if tts_provider == "cartesia":  # legacy alias
        tts_provider = "together"
    _add(tts_provider)

    return keys


def validate_env(cfg: dict[str, Any]) -> list[str]:
    """Return env var names that are required but unset (sorted, deduped)."""
    return sorted(k for k in required_env_keys(cfg) if not os.environ.get(k))


# ── Transcription client ────────────────────────────────────

def _parse_transcription_config(cfg: dict[str, Any]) -> tuple[str, str]:
    """Return (provider, model) from the models.transcription config entry."""
    entry = cfg["models"]["transcription"]
    if isinstance(entry, dict):
        return entry.get("provider", "together"), entry["model"]
    return "together", entry


def get_transcription_model(cfg: dict[str, Any]) -> str:
    _, model = _parse_transcription_config(cfg)
    return model


def get_transcription_provider(cfg: dict[str, Any]) -> str:
    provider, _ = _parse_transcription_config(cfg)
    return provider


def make_transcription_client(
    cfg: dict[str, Any],
    *,
    together_key_override: str | None = None,
    openai_key_override: str | None = None,
):
    """Create the appropriate Whisper client based on the transcription provider config.

    The resulting client exposes `audio.transcriptions.create(...)` — both the
    Together and OpenAI SDKs share that surface, so the transcriber code does
    not need to branch. For provider="faster-whisper", we return a local
    adapter exposing the same surface (see pipeline.transcriber_local).
    """
    provider, _ = _parse_transcription_config(cfg)

    if provider == "faster-whisper":
        # Lazy-import to avoid hard dep when user is on cloud-only stack.
        from .transcriber_local import FasterWhisperAdapter
        model_name = get_transcription_model(cfg)
        tcfg = cfg.get("transcription", {})
        return FasterWhisperAdapter(
            model_size=model_name,
            device=tcfg.get("local_device", "auto"),
            compute_type=tcfg.get("local_compute_type", "auto"),
        )

    if provider == "openai":
        from openai import OpenAI
        api_key = openai_key_override or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        return OpenAI(api_key=api_key)

    from together import Together
    api_key = together_key_override or os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY environment variable is not set.")
    return Together(api_key=api_key)
