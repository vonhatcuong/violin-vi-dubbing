import json
from types import SimpleNamespace

import pytest

from pipeline import config as pipeline_config
from pipeline import translator


class FakeClient:
    """Minimal OpenAI-SDK-shaped client that records kwargs and returns canned JSON."""

    def __init__(self, content: str):
        self.calls: list[dict] = []
        self._content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        msg = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


@pytest.fixture
def cfg():
    return pipeline_config.load()


def test_default_uses_json_schema(cfg, monkeypatch):
    monkeypatch.setitem(cfg["translation"], "response_format", "json_schema")
    client = FakeClient(json.dumps({"translations": ["xin chào"]}))
    out = translator._try_batch(["hello"], "Vietnamese", "English", client)
    assert out == ["xin chào"]
    assert client.calls[0]["response_format"]["type"] == "json_schema"


def test_json_object_mode(cfg, monkeypatch):
    monkeypatch.setitem(cfg["translation"], "response_format", "json_object")
    client = FakeClient(json.dumps({"translations": ["xin chào"]}))
    translator._try_batch(["hello"], "Vietnamese", "English", client)
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_no_think_prefix_for_local_providers(cfg, monkeypatch):
    monkeypatch.setitem(cfg["models"], "translation", {"provider": "openai_compat", "model": "m", "base_url": "http://x"})
    client = FakeClient(json.dumps({"translations": ["xin chào"]}))
    translator._try_batch(["hello"], "Vietnamese", "English", client)
    system_msg = client.calls[0]["messages"][0]["content"]
    assert system_msg.startswith("/no_think")
