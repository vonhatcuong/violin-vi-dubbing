import json
from types import SimpleNamespace

import pytest

from pipeline import config as pipeline_config
from pipeline import translator


class FakeClient:
    def __init__(self, content):
        self.calls = []
        self._content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))], usage=None)


@pytest.fixture
def cfg():
    return pipeline_config.load()


def _set_provider(cfg, provider, **extra):
    cfg["models"]["translation"] = {"provider": provider, "model": "m", **extra}


def test_ollama_sends_reasoning_effort_none_by_default(cfg):
    _set_provider(cfg, "ollama")
    client = FakeClient(json.dumps({"translations": ["xin chào"]}))
    translator._try_batch(["hello"], "Vietnamese", "English", client)
    assert client.calls[0]["extra_body"] == {"reasoning_effort": "none"}


def test_reasoning_effort_is_configurable(cfg, monkeypatch):
    _set_provider(cfg, "ollama")
    monkeypatch.setitem(cfg["translation"], "reasoning_effort", "low")
    client = FakeClient(json.dumps({"translation": "xin chào"}))
    translator._translate_single("hello", "Vietnamese", "English", client)
    assert client.calls[0]["extra_body"] == {"reasoning_effort": "low"}


def test_openai_compat_sends_nothing_unless_configured(cfg, monkeypatch):
    _set_provider(cfg, "openai_compat", base_url="http://x")
    monkeypatch.delitem(cfg["translation"], "reasoning_effort")
    client = FakeClient(json.dumps({"translations": ["xin chào"]}))
    translator._try_batch(["hello"], "Vietnamese", "English", client)
    assert "extra_body" not in client.calls[0]
    monkeypatch.setitem(cfg["translation"], "reasoning_effort", "none")
    translator._try_batch(["hello"], "Vietnamese", "English", client)
    assert client.calls[1]["extra_body"] == {"reasoning_effort": "none"}


def test_together_keeps_enable_thinking_false(cfg):
    _set_provider(cfg, "together")
    client = FakeClient(json.dumps({"translations": ["xin chào"]}))
    translator._try_batch(["hello"], "Vietnamese", "English", client)
    assert client.calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_shorten_segment_uses_provider_extra(cfg):
    _set_provider(cfg, "ollama")
    client = FakeClient(json.dumps({"translation": "ngắn"}))
    translator.shorten_segment("long source", "bản dịch dài", 3, 1.0, "Vietnamese", client)
    assert client.calls[0]["extra_body"] == {"reasoning_effort": "none"}
