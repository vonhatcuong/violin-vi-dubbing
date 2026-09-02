import pytest

from pipeline import llm_client


def _cfg(translation):
    return {
        "models": {
            "transcription": {"provider": "faster-whisper", "model": "large-v3-turbo"},
            "translation": translation,
            "chat": translation,
            "tts": {"provider": "f5vi", "model": "f5-tts-vi"},
        }
    }


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("OLLAMA_API_KEY", "OLLAMA_BASE_URL", "OPENAI_COMPAT_BASE_URL", "OPENAI_COMPAT_API_KEY"):
        monkeypatch.delenv(k, raising=False)


def test_local_ollama_needs_no_api_key():
    client = llm_client.make_translation_client(_cfg({"provider": "ollama", "model": "qwen3.5:9b-mlx"}))
    assert str(client.base_url).rstrip("/") == "http://localhost:11434/v1"


def test_config_base_url_wins():
    client = llm_client.make_translation_client(
        _cfg({"provider": "ollama", "model": "m", "base_url": "http://gpubox:11434/v1"})
    )
    assert str(client.base_url).rstrip("/") == "http://gpubox:11434/v1"


def test_ollama_cloud_without_key_raises():
    with pytest.raises(RuntimeError):
        llm_client.make_translation_client(
            _cfg({"provider": "ollama", "model": "m", "base_url": "https://ollama.com/v1"})
        )


def test_openai_compat_requires_base_url():
    with pytest.raises(RuntimeError):
        llm_client.make_translation_client(_cfg({"provider": "openai_compat", "model": "m"}))


def test_openai_compat_uses_base_url_and_placeholder_key():
    client = llm_client.make_translation_client(
        _cfg({"provider": "openai_compat", "model": "m", "base_url": "http://vllm:8000/v1"})
    )
    assert str(client.base_url).rstrip("/") == "http://vllm:8000/v1"


def test_validate_env_empty_for_local_stack():
    assert llm_client.validate_env(_cfg({"provider": "ollama", "model": "m"})) == []
