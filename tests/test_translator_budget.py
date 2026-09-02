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


@pytest.fixture(autouse=True)
def _cfg():
    pipeline_config.load()


def test_batch_prompt_annotates_budget():
    client = FakeClient(json.dumps({"translations": ["xin chào"]}))
    translator._try_batch(["hello everyone"], "Vietnamese", "English", client, budgets=[(3.2, 15)])
    user_msg = client.calls[0]["messages"][1]["content"]
    assert "[0] (3.2s, ≤15 syllables)" in user_msg
    assert "syllable budget" in user_msg.lower()


def test_batch_prompt_without_budget_is_unchanged():
    client = FakeClient(json.dumps({"translations": ["xin chào"]}))
    translator._try_batch(["hello"], "Vietnamese", "English", client)
    user_msg = client.calls[0]["messages"][1]["content"]
    assert "syllables)" not in user_msg


def test_budget_prompt_replaces_shorter_rule():
    client = FakeClient(json.dumps({"translations": ["xin chào"]}))
    translator._try_batch(["hello everyone"], "Vietnamese", "English", client, budgets=[(3.2, 15)])
    user_msg = client.calls[0]["messages"][1]["content"]
    assert "MUST be SHORTER" not in user_msg
    assert "80–95%" in user_msg or "80-95%" in user_msg


def test_no_budget_prompt_keeps_shorter_rule():
    client = FakeClient(json.dumps({"translations": ["xin chào"]}))
    translator._try_batch(["hello"], "Vietnamese", "English", client)
    user_msg = client.calls[0]["messages"][1]["content"]
    assert "MUST be SHORTER" in user_msg


def test_shorten_segment_returns_translation():
    client = FakeClient(json.dumps({"translation": "chào mọi người"}))
    out = translator.shorten_segment("hello everyone, welcome", "xin chào tất cả mọi người, chào mừng", 4, 1.0, "Vietnamese", client)
    assert out == "chào mọi người"
    user_msg = client.calls[0]["messages"][1]["content"]
    assert "at most 4 syllables" in user_msg


def test_shorten_segment_falls_back_on_bad_json():
    client = FakeClient("not json")
    out = translator.shorten_segment("src", "hiện tại", 3, 1.0, "Vietnamese", client)
    assert out == "hiện tại"
