import json
import threading
import time
from types import SimpleNamespace

from pipeline import config as pipeline_config
from pipeline import translator
from pipeline.transcriber import Segment


class SlowFakeClient:
    """Each call sleeps 0.2 s and echoes numbered translations; records max concurrency."""

    def __init__(self):
        self.active = 0; self.max_active = 0; self.lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        with self.lock:
            self.active += 1; self.max_active = max(self.max_active, self.active)
        time.sleep(0.2)
        user = kwargs["messages"][1]["content"]
        n = user.count("\n[")  # numbered lines
        with self.lock:
            self.active -= 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"translations": [f"vi{i}" for i in range(n)]})))], usage=None)


def test_parallel_batches_preserve_order_and_run_concurrently(monkeypatch):
    cfg = pipeline_config.load()
    monkeypatch.setitem(cfg["translation"], "batch_size", 2)
    monkeypatch.setitem(cfg["translation"], "parallel_batches", 3)
    segs = [Segment(id=i, start=i, end=i + 1, text=f"s{i}") for i in range(6)]
    client = SlowFakeClient()
    t0 = time.time()
    out = translator.translate_segments(segs, "Vietnamese", client)
    assert [s.id for s in out] == list(range(6)) and [s.source_text for s in out] == [f"s{i}" for i in range(6)]
    assert client.max_active >= 2
    assert time.time() - t0 < 0.55           # 3 batches in parallel, not 0.6 s serial


def test_parallel_batches_default_is_serial(monkeypatch):
    cfg = pipeline_config.load()
    monkeypatch.setitem(cfg["translation"], "batch_size", 2)
    segs = [Segment(id=i, start=i, end=i + 1, text=f"s{i}") for i in range(4)]
    client = SlowFakeClient()
    translator.translate_segments(segs, "Vietnamese", client)
    assert client.max_active == 1
