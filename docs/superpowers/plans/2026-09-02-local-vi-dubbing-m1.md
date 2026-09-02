# Local EN→VI Dubbing — M1 (single-speaker, fully local) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chạy `uv run main.py lecture.mp4 out_vi.mp4 --language Vietnamese --config config/local_mac.yaml` hoàn toàn offline: faster-whisper → Ollama local → F5-TTS-Vietnamese → vòng lặp fit-duration → merger sẵn có, với giọng Việt cố định từ voice bank.

**Architecture:** Giữ nguyên `dub_video` 5 bước của violin; thêm backend TTS `f5vi`, provider LLM local (`ollama` localhost / `openai_compat`), module `fitter` chèn giữa dịch và merge (ước lượng theo âm tiết → rút gọn bằng LLM → TTS đo → re-synth với speed ≤ 1.15 → mượn pause bằng cách kéo `seg.end`), phần dư giao merger (video ≤ 8 %, atempo ≤ 1.4, hard trim). Mọi tính năng mới tắt trong `config/default.yaml`; bật trong `config/local_mac.yaml`.

**Tech Stack:** Python 3.13 + uv, ffmpeg, faster-whisper (đã có), `openai` SDK trỏ Ollama `http://localhost:11434/v1` (model `qwen3.5:9b-mlx`), `f5-tts` 1.1.x (PyTorch, MPS/CUDA) + checkpoint `hynt/F5-TTS-Vietnamese-ViVoice`, `vinorm` 2.0.7 (fallback `num2words`), `soundfile`, `pyyaml`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-local-vi-dubbing-design.md`

## Global Constraints

- Python `>=3.10` (repo chạy 3.13 qua `.python-version`); quản lý bằng `uv`; chạy test bằng `uv run python -m pytest -q` (KHÔNG `uv run pytest`, pytest hệ thống 3.14 không thấy `pipeline`).
- 35 test hiện có phải pass nguyên trạng sau mỗi task; mọi khoá config mới có mặc định = tắt/an toàn trong `config/default.yaml`.
- Không gọi dịch vụ mạng trong đường local: không Edge-TTS, không Ollama Cloud, không Together/OpenAI khi dùng `config/local_mac.yaml`.
- Không bao giờ làm chậm giọng (speed < 1.0); `fit.max_tts_speed` = 1.15; merger giữ `speed_clamp_min: 0.92` (video ≤ 8 %), `max_audio_speedup: 1.4`.
- F5-TTS-Vietnamese-ViVoice: CC-BY-NC-SA-4.0 — chỉ dùng cá nhân/nghiên cứu; model nhận text **lowercase**, NFC.
- Style code: giữ pattern module backend TTS như `pipeline/tts_supertonic.py` (hàm module-level, shared instance + lock), config qua `pipeline.config.get()`, in tiến độ bằng `print("      …")`.
- Commit sau mỗi task, message tiếng Anh theo prefix `feat|fix|test|docs|chore(scope):`, kết thúc bằng
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` và `Claude-Session: https://claude.ai/code/session_01HjBXyTtKKvNCAwEkCRHgNs`.

---

## File map (M1)

| File | Trách nhiệm |
|---|---|
| `pyproject.toml` (modify) | dev group `pytest`; extras `local-mac` / `local-gpu` (`local` = alias) |
| `pipeline/transcriber.py` (modify) | `Segment.source_text`; `merge_continuous_segments(min_duration=)` gộp đơn vị quá ngắn |
| `pipeline/translator.py` (modify) | giữ `speaker`/`source_text`; `response_format` cấu hình; budget theo slot trong batch prompt; `shorten_segment()` |
| `prompts/translate.yaml` (modify) | `{budget_block}` trong `batch_user`; `shorten_system` / `shorten_user` |
| `pipeline/llm_client.py` (modify) | `ollama` local mặc định không cần key; provider `openai_compat` (vLLM); `f5vi` không cần env |
| `pipeline/devices.py` (create) | `pick_device()`, `free_memory()` |
| `pipeline/vi_text.py` (create) | `normalize_for_tts()`, `count_syllables()` |
| `pipeline/voices.py` (create) + `assets/voices/vi/catalog.yaml` (create) + `scripts/make_ref_clip.py` (create) | voice bank giọng Việt cố định (clip mẫu + ref text) |
| `pipeline/tts_f5vi.py` (create), `pipeline/tts.py` (modify) | backend F5-TTS-vi; `make_synthesizer()` cho fitter |
| `pipeline/fitter.py` (create) | `DubUnit`, `compute_slots`, `fit_text`, `fit_audio`, `apply_units`, `save_units` |
| `pipeline/merger.py` (modify) | `merge_video.hard_trim` (cắt + fade thay freeze) |
| `pipeline/orchestrator.py`, `main.py`, `config/default.yaml`, `config/local_mac.yaml`, `config/local_gpu.yaml` (create), `resume_from_segments.py` (modify) | nối stage fit vào pipeline, cờ CLI, preset |
| `tests/test_translator_segments.py`, `tests/test_llm_client_local.py`, `tests/test_vi_text.py`, `tests/test_devices.py`, `tests/test_voices.py`, `tests/test_tts_f5vi.py`, `tests/test_transcriber_min_duration.py`, `tests/test_fitter.py`, `tests/test_merger_hard_trim.py`, `tests/test_orchestrator_fit.py` (create) | test theo từng task |

---

### Task 1: Dev tooling, `Segment.source_text`, sửa lỗi mất `speaker` khi dịch

**Files:**
- Modify: `pyproject.toml` (thêm dependency group)
- Modify: `pipeline/transcriber.py:81-86` (`Segment`)
- Modify: `pipeline/translator.py:297-300` (`translate_segments` return)
- Modify: `resume_from_segments.py:52-69` (`load_segments`)
- Test: `tests/test_translator_segments.py`

**Interfaces:**
- Produces: `Segment(id, start, end, text, speaker="SPEAKER_00", source_text="")` — mọi task sau dùng `source_text` làm câu gốc tiếng Anh của bản dịch.

- [ ] **Step 1: Thêm dev group pytest**

Thêm vào cuối `pyproject.toml`:

```toml
[dependency-groups]
dev = ["pytest>=8.0"]
```

Chạy `uv sync --extra local` (uv cài group `dev` mặc định). Kiểm tra: `uv run python -c "import pytest; print(pytest.__version__)"`.

- [ ] **Step 2: Viết test thất bại**

`tests/test_translator_segments.py`:

```python
from unittest.mock import patch

from pipeline.transcriber import Segment
from pipeline.translator import translate_segments


def test_translate_segments_keeps_speaker_and_source_text():
    segs = [Segment(id=0, start=0.0, end=1.0, text="Hello there", speaker="SPEAKER_01")]
    with patch("pipeline.translator._translate_batch", return_value=["Xin chào"]):
        out = translate_segments(segs, "Vietnamese", client=object())
    assert out[0].text == "Xin chào"
    assert out[0].speaker == "SPEAKER_01"
    assert out[0].source_text == "Hello there"


def test_segment_source_text_defaults_empty():
    assert Segment(id=0, start=0.0, end=1.0, text="x").source_text == ""
```

- [ ] **Step 3: Chạy test, xác nhận thất bại**

Run: `uv run python -m pytest tests/test_translator_segments.py -v`
Expected: FAIL — `TypeError: Segment.__init__() got an unexpected keyword argument 'source_text'` / `AttributeError: 'Segment' object has no attribute 'source_text'`.

- [ ] **Step 4: Sửa `Segment` và `translate_segments`**

`pipeline/transcriber.py` (dataclass `Segment`):

```python
@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    speaker: str = "SPEAKER_00"
    source_text: str = ""   # original-language text once `text` holds a translation
```

`pipeline/translator.py` cuối `translate_segments`:

```python
    return [
        Segment(
            id=s.id, start=s.start, end=s.end, text=t,
            speaker=s.speaker, source_text=s.text,
        )
        for s, t in zip(segments, translated_texts)
    ]
```

`resume_from_segments.py` trong `load_segments`, thêm vào constructor:

```python
            speaker=s.get("speaker", "SPEAKER_00"),
            source_text=s.get("source_text", ""),
```

- [ ] **Step 5: Chạy toàn bộ test**

Run: `uv run python -m pytest -q`
Expected: `37 passed` (35 cũ + 2 mới).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock pipeline/transcriber.py pipeline/translator.py resume_from_segments.py tests/test_translator_segments.py
git commit -m "fix(translator): keep speaker and source_text on translated segments; add pytest dev group"
```

---

### Task 2: LLM client local — Ollama localhost không cần key, provider `openai_compat`

**Files:**
- Modify: `pipeline/llm_client.py:14-16, 48-96, 119-141, 146-155`
- Test: `tests/test_llm_client_local.py`

**Interfaces:**
- Consumes: `cfg["models"]["translation"]` dict `{provider, model, base_url?, api_key?}`.
- Produces: `make_translation_client(cfg)` trả `openai.OpenAI` trỏ `base_url` local; `validate_env(cfg) == []` cho stack local; `_PROVIDER_ENV_KEY` có `ollama: None`, `openai_compat: None`, `f5vi: None`.

- [ ] **Step 1: Viết test thất bại**

`tests/test_llm_client_local.py`:

```python
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
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `uv run python -m pytest tests/test_llm_client_local.py -v`
Expected: FAIL — `RuntimeError: OLLAMA_API_KEY is not set` và `validate_env` trả `['OLLAMA_API_KEY']`.

- [ ] **Step 3: Sửa `pipeline/llm_client.py`**

Thay hằng số và `_make_openai_compat_client`:

```python
# Ollama: local daemon by default; Ollama Cloud only when base_url points there.
_OLLAMA_LOCAL_BASE_URL = "http://localhost:11434/v1"
_OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"


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
```

Trong `make_translation_client`:

```python
    if provider in _OPENAI_COMPAT_PROVIDERS:
        return _make_openai_compat_client(
            provider,
            entry=_entry_dict(cfg, "translation"),
            openai_key_override=openai_key_override,
            ollama_key_override=ollama_key_override,
        )
```

Trong `make_chat_client` tương tự với `entry=_entry_dict(cfg, "chat")`.

`_PROVIDER_ENV_KEY`:

```python
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
}
```

Cập nhật docstring `get_translation_provider` thành `'openai', 'together', 'ollama' or 'openai_compat'`.

- [ ] **Step 4: Chạy test**

Run: `uv run python -m pytest tests/test_llm_client_local.py -q && uv run python -m pytest -q`
Expected: 6 passed; tổng `43 passed`.

- [ ] **Step 5: Commit**

```bash
git add pipeline/llm_client.py tests/test_llm_client_local.py
git commit -m "feat(llm): local Ollama without API key and openai_compat provider"
```

---

### Task 3: `translation.response_format` cấu hình (json_schema | json_object) + `/no_think` cho `openai_compat`

**Files:**
- Modify: `pipeline/translator.py:119-142, 189-215`
- Modify: `config/default.yaml:37-40` (khối `translation`)
- Test: `tests/test_translator_response_format.py`

**Interfaces:**
- Produces: `translator._response_format(name: str, schema: dict) -> dict` dùng bởi mọi lời gọi LLM trong translator (kể cả `shorten_segment` ở Task 7); `translator._is_local_provider(cfg) -> bool`.

- [ ] **Step 1: Viết test thất bại**

`tests/test_translator_response_format.py`:

```python
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
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `uv run python -m pytest tests/test_translator_response_format.py -v`
Expected: `test_json_object_mode` FAIL (vẫn gửi `json_schema`), `test_no_think_prefix_for_local_providers` FAIL.

- [ ] **Step 3: Sửa `pipeline/translator.py`**

Thêm sau `SINGLE_SCHEMA`:

```python
def _response_format(name: str, schema: dict) -> dict:
    """Build the `response_format` kwarg per `translation.response_format`.

    json_schema (default) — strict schema; supported by OpenAI/Together.
    json_object            — plain JSON mode; safer for Ollama/vLLM, which have
                             hung on strict json_schema (see commit 3957df5).
    """
    mode = _tcfg().get("response_format", "json_schema")
    if mode == "json_object":
        return {"type": "json_object"}
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


def _is_local_provider(cfg: dict) -> bool:
    """Local LLM servers (Ollama, vLLM/llama.cpp via openai_compat) get the /no_think switch."""
    return get_translation_provider(cfg) in ("ollama", "openai_compat")
```

Trong `_translate_single`: thay `if get_translation_provider(cfg) == "ollama":` bằng `if _is_local_provider(cfg):` và thay khối `response_format={...}` bằng `response_format=_response_format("single_translation", SINGLE_SCHEMA),`.

Trong `_try_batch`: tương tự — `if _is_local_provider(cfg):` và `response_format=_response_format("translation_response", BATCH_SCHEMA),`.

`config/default.yaml` khối `translation` thêm dòng:

```yaml
  response_format: json_schema   # json_schema (OpenAI/Together) | json_object (Ollama, vLLM — plain JSON mode)
```

- [ ] **Step 4: Chạy test**

Run: `uv run python -m pytest -q`
Expected: `46 passed`.

- [ ] **Step 5: Commit**

```bash
git add pipeline/translator.py config/default.yaml tests/test_translator_response_format.py
git commit -m "feat(translator): configurable response_format and /no_think for local providers"
```

---

### Task 4: `pipeline/vi_text.py` — chuẩn hoá text tiếng Việt cho TTS + đếm âm tiết

**Files:**
- Create: `pipeline/vi_text.py`
- Test: `tests/test_vi_text.py`

**Interfaces:**
- Produces: `normalize_for_tts(text: str, *, use_vinorm: bool | None = None, loanwords: dict[str, str] | None = None) -> str` (lowercase, NFC, số → chữ, `%` → "phần trăm", loanword map, bỏ ký tự ngoài bảng chữ + dấu câu cơ bản, gộp khoảng trắng); `count_syllables(text: str) -> int` (đếm token chữ/số sau normalize).
- Consumes: package `vinorm` (tuỳ chọn), `num2words` (fallback).

- [ ] **Step 1: Viết test thất bại**

`tests/test_vi_text.py`:

```python
from pipeline import vi_text


def test_lowercase_and_nfc():
    # "Chào" written with combining accent must collapse to precomposed lowercase
    decomposed = "Chào"
    assert vi_text.normalize_for_tts(f"Xin {decomposed} bạn", use_vinorm=False) == "xin chào bạn"


def test_numbers_become_vietnamese_words():
    out = vi_text.normalize_for_tts("có 23 người", use_vinorm=False)
    assert out == "có hai mươi ba người"


def test_percent_and_decimal():
    out = vi_text.normalize_for_tts("tăng 2.5%", use_vinorm=False)
    assert out == "tăng hai phẩy năm phần trăm"


def test_loanword_map_is_case_insensitive_and_word_bounded():
    out = vi_text.normalize_for_tts("dùng GPU và gpus", use_vinorm=False, loanwords={"GPU": "gi pi u"})
    assert out == "dùng gi pi u và gpus"


def test_unsupported_symbols_are_dropped():
    out = vi_text.normalize_for_tts("a → b “c”", use_vinorm=False)
    assert "→" not in out and "“" not in out
    assert out == "a b c"


def test_count_syllables_after_normalization():
    assert vi_text.count_syllables("Xin chào các bạn, 2 người!") == 6
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `uv run python -m pytest tests/test_vi_text.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.vi_text'`.

- [ ] **Step 3: Cài fallback và viết module**

`uv add num2words` (thêm vào dependencies chính — nhẹ, pure Python).

`pipeline/vi_text.py`:

```python
"""Vietnamese text frontend for local TTS (F5-TTS-Vietnamese expects lowercase NFC text).

normalize_for_tts:  vinorm (if installed & enabled) → loanword map → numbers/percent
                    → lowercase → drop unsupported symbols → collapse spaces.
count_syllables:    Vietnamese is monosyllabic and space-delimited, so after
                    normalization every alphanumeric token is one syllable.
"""

from __future__ import annotations

import re
import unicodedata

try:  # optional: vinorm ships a native binary that may not exist on every platform
    from vinorm import TTSnorm as _vinorm_norm
except Exception:  # pragma: no cover - depends on platform
    _vinorm_norm = None

from num2words import num2words

_VI_LETTERS = "a-zA-ZÀ-ỹ"
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_PERCENT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
_TOKEN_RE = re.compile(rf"[{_VI_LETTERS}0-9]+")
_ALLOWED_RE = re.compile(rf"[^{_VI_LETTERS}0-9\s.,!?;:'\-]")
_SPACES_RE = re.compile(r"\s+")


def _number_to_words(token: str) -> str:
    text = token.replace(",", ".")
    try:
        value = float(text) if "." in text else int(text)
        return num2words(value, lang="vi")
    except Exception:
        return token


def _expand_numbers(text: str) -> str:
    text = _PERCENT_RE.sub(lambda m: f"{_number_to_words(m.group(1))} phần trăm", text)
    return _NUMBER_RE.sub(lambda m: _number_to_words(m.group(0)), text)


def _apply_loanwords(text: str, loanwords: dict[str, str] | None) -> str:
    for src, dst in (loanwords or {}).items():
        text = re.sub(rf"(?<![{_VI_LETTERS}0-9]){re.escape(src)}(?![{_VI_LETTERS}0-9])", dst, text, flags=re.IGNORECASE)
    return text


def normalize_for_tts(
    text: str,
    *,
    use_vinorm: bool | None = None,
    loanwords: dict[str, str] | None = None,
) -> str:
    """Return TTS-ready Vietnamese text: NFC, lowercase, numbers spelled out."""
    text = unicodedata.normalize("NFC", text or "")
    text = _apply_loanwords(text, loanwords)
    want_vinorm = (_vinorm_norm is not None) if use_vinorm is None else use_vinorm
    if want_vinorm and _vinorm_norm is not None:
        try:
            text = _vinorm_norm(text, punc=False, unknown=False, lower=True, rule=False)
        except Exception:
            pass  # fall through to the built-in expansion
    text = _expand_numbers(text)
    text = text.lower()
    text = _ALLOWED_RE.sub(" ", text)
    text = _SPACES_RE.sub(" ", text).strip()
    return text


def count_syllables(text: str) -> int:
    """Count spoken syllables ≈ alphanumeric tokens after normalization."""
    return len(_TOKEN_RE.findall(normalize_for_tts(text, use_vinorm=False)))
```

- [ ] **Step 4: Chạy test**

Run: `uv run python -m pytest tests/test_vi_text.py -v`
Expected: 6 passed. Nếu `test_numbers_become_vietnamese_words` khác ở dạng "hai mươi ba" (num2words vi có thể trả "hai mươi ba"), giữ nguyên test; nếu num2words trả biến thể khác, sửa expected theo output thực và ghi chú trong test.

- [ ] **Step 5: Commit**

```bash
git add pipeline/vi_text.py tests/test_vi_text.py pyproject.toml uv.lock
git commit -m "feat(vi_text): Vietnamese TTS text normalization and syllable count"
```

---

### Task 5: `devices.py`, voice bank `voices.py`, catalog và script `make_ref_clip.py`

**Files:**
- Create: `pipeline/devices.py`, `pipeline/voices.py`, `assets/voices/vi/catalog.yaml`, `assets/voices/vi/README.md`, `scripts/make_ref_clip.py`
- Modify: `config/default.yaml` (khối `voices`)
- Test: `tests/test_devices.py`, `tests/test_voices.py`

**Interfaces:**
- Produces: `devices.pick_device(requested="auto") -> str`, `devices.free_memory() -> None`;
  `voices.Voice(name, gender, ref_wav: Path, ref_text, description)`, `voices.load_catalog(bank: Path | None = None) -> dict[str, Voice]`, `voices.get_voice(name, bank=None) -> Voice`, `voices.native_voices_for(language_code, bank=None) -> list[str]` (`[male, female]`), `voices.assign_voices(speakers, default_voice, voice_map=None, genders=None, bank=None) -> dict[str, str]`.
- Config: `voices.bank: assets/voices/vi`, `voices.default_male: ""`, `voices.default_female: ""`.

- [ ] **Step 1: Viết test thất bại**

`tests/test_devices.py`:

```python
from pipeline import devices


def test_explicit_device_is_returned_verbatim():
    assert devices.pick_device("cpu") == "cpu"
    assert devices.pick_device("cuda") == "cuda"


def test_auto_picks_a_known_device():
    assert devices.pick_device("auto") in {"cpu", "mps", "cuda"}


def test_free_memory_never_raises():
    devices.free_memory()
```

`tests/test_voices.py`:

```python
from pathlib import Path

import pytest
import yaml

from pipeline import voices


@pytest.fixture
def bank(tmp_path: Path) -> Path:
    (tmp_path / "nam-1.wav").write_bytes(b"RIFF")
    (tmp_path / "nu-1.wav").write_bytes(b"RIFF")
    (tmp_path / "catalog.yaml").write_text(yaml.safe_dump({"voices": [
        {"name": "nam-1", "gender": "male", "wav": "nam-1.wav", "ref_text": "xin chào", "description": "nam bắc"},
        {"name": "nu-1", "gender": "female", "wav": "nu-1.wav", "ref_text": "xin chào", "description": "nữ bắc"},
    ]}), encoding="utf-8")
    return tmp_path


def test_load_catalog_resolves_paths(bank):
    cat = voices.load_catalog(bank)
    assert set(cat) == {"nam-1", "nu-1"}
    assert cat["nam-1"].ref_wav == bank / "nam-1.wav"
    assert cat["nu-1"].gender == "female"


def test_native_voices_first_male_then_female(bank):
    assert voices.native_voices_for("vi", bank=bank) == ["nam-1", "nu-1"]


def test_missing_bank_raises_helpful_error(tmp_path):
    with pytest.raises(RuntimeError, match="make_ref_clip"):
        voices.native_voices_for("vi", bank=tmp_path)


def test_assign_voices_uses_map_then_gender_then_default(bank):
    out = voices.assign_voices(
        ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
        default_voice="nam-1",
        voice_map={"SPEAKER_00": "nu-1"},
        genders={"SPEAKER_01": "female"},
        bank=bank,
    )
    assert out == {"SPEAKER_00": "nu-1", "SPEAKER_01": "nu-1", "SPEAKER_02": "nam-1"}


def test_get_voice_unknown_name(bank):
    with pytest.raises(KeyError):
        voices.get_voice("nope", bank=bank)
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `uv run python -m pytest tests/test_devices.py tests/test_voices.py -v`
Expected: FAIL — `ModuleNotFoundError` cho `pipeline.devices` và `pipeline.voices`.

- [ ] **Step 3: Viết `pipeline/devices.py`**

```python
"""Device selection + memory release shared by local model backends."""

from __future__ import annotations

import gc


def pick_device(requested: str = "auto") -> str:
    """Return 'cuda' | 'mps' | 'cpu'. Explicit values pass through unchanged."""
    if requested and requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def free_memory() -> None:
    """Best-effort release of accelerator memory between pipeline stages."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
```

- [ ] **Step 4: Viết `pipeline/voices.py`**

```python
"""Fixed Vietnamese voice bank for zero-shot TTS backends (F5-TTS-vi).

A voice = reference clip (5–10 s, 24 kHz mono WAV) + its exact transcript.
Catalog lives in `<bank>/catalog.yaml`:

    voices:
      - name: nam-1
        gender: male          # male | female
        wav: nam-1.wav        # relative to the bank directory
        ref_text: "câu đúng nội dung clip, viết thường"
        description: "Giọng nam miền Bắc, trầm"

Create entries with `uv run scripts/make_ref_clip.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from . import config as _conf

_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Voice:
    name: str
    gender: str
    ref_wav: Path
    ref_text: str
    description: str = ""


def bank_dir(bank: Path | None = None) -> Path:
    if bank is not None:
        return Path(bank)
    configured = _conf.get().get("voices", {}).get("bank", "assets/voices/vi")
    p = Path(configured).expanduser()
    return p if p.is_absolute() else _REPO_ROOT / p


def load_catalog(bank: Path | None = None) -> dict[str, Voice]:
    root = bank_dir(bank)
    catalog = root / "catalog.yaml"
    if not catalog.exists():
        return {}
    data = yaml.safe_load(catalog.read_text(encoding="utf-8")) or {}
    out: dict[str, Voice] = {}
    for entry in data.get("voices", []) or []:
        out[entry["name"]] = Voice(
            name=entry["name"],
            gender=entry.get("gender", "male"),
            ref_wav=root / entry["wav"],
            ref_text=entry["ref_text"],
            description=entry.get("description", ""),
        )
    return out


def get_voice(name: str, bank: Path | None = None) -> Voice:
    catalog = load_catalog(bank)
    if name not in catalog:
        raise KeyError(f"voice '{name}' not in bank {bank_dir(bank)} (have: {sorted(catalog)})")
    return catalog[name]


def native_voices_for(language_code: str, bank: Path | None = None) -> list[str]:
    """Return [default_male, default_female] names from the bank."""
    _ = language_code  # bank is per-language by directory; only `vi` shipped for now
    catalog = load_catalog(bank)
    if not catalog:
        raise RuntimeError(
            f"Voice bank {bank_dir(bank)} is empty. Add a reference clip with\n"
            "  uv run scripts/make_ref_clip.py --source clip.wav --start 0 --end 8 --name nam-1 --gender male"
        )
    vcfg = _conf.get().get("voices", {})

    def _first(gender: str, preferred: str) -> str:
        if preferred and preferred in catalog:
            return preferred
        for v in catalog.values():
            if v.gender == gender:
                return v.name
        return next(iter(catalog))

    return [_first("male", vcfg.get("default_male", "")), _first("female", vcfg.get("default_female", ""))]


def all_voices(bank: Path | None = None) -> dict[str, list[str]]:
    return {"vi": list(load_catalog(bank))}


def voice_descriptions(bank: Path | None = None) -> dict[str, str]:
    return {v.name: f"{v.gender} — {v.description}" for v in load_catalog(bank).values()}


def assign_voices(
    speakers: list[str],
    default_voice: str,
    voice_map: dict[str, str] | None = None,
    genders: dict[str, str] | None = None,
    bank: Path | None = None,
) -> dict[str, str]:
    """speaker → voice name. Priority: explicit map > detected gender > default."""
    voice_map = voice_map or {}
    genders = genders or {}
    male, female = native_voices_for("vi", bank=bank)
    out: dict[str, str] = {}
    for spk in speakers:
        if spk in voice_map:
            out[spk] = voice_map[spk]
        elif genders.get(spk) == "female":
            out[spk] = female
        elif genders.get(spk) == "male":
            out[spk] = male
        else:
            out[spk] = default_voice
    return out
```

- [ ] **Step 5: Catalog rỗng, README và script tạo clip**

`assets/voices/vi/catalog.yaml`:

```yaml
# Voice bank for fixed Vietnamese voices. Add entries with scripts/make_ref_clip.py.
# Reference clips must be audio you have the right to use (your own recording, CC0…).
voices: []
```

`assets/voices/vi/README.md`:

```markdown
# Voice bank (vi)

Mỗi giọng = 1 clip WAV 24 kHz mono dài 5–10 s, nói rõ, không nhạc nền + transcript chính xác (viết thường).
Tạo bằng:

    uv run scripts/make_ref_clip.py --source path/to/audio_or_video --start 12.0 --end 20.0 --name nam-1 --gender male
    # thêm --ref-text "..." nếu không muốn tự transcribe bằng faster-whisper

`catalog.yaml` được cập nhật tự động. Không commit clip có bản quyền của người khác.
```

`scripts/make_ref_clip.py`:

```python
"""Cut a reference clip for the voice bank and register it in catalog.yaml.

    uv run scripts/make_ref_clip.py --source talk.mp4 --start 12 --end 20 --name nam-1 --gender male
    uv run scripts/make_ref_clip.py --source me.wav --start 0 --end 8 --name nu-1 --gender female --ref-text "..."

Without --ref-text the clip is transcribed locally with faster-whisper (language vi).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.ffmpeg_utils import FFMPEG_EXE  # noqa: E402
from pipeline.vi_text import normalize_for_tts  # noqa: E402
from pipeline.voices import bank_dir  # noqa: E402


def _transcribe_vi(path: Path) -> str:
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(path), language="vi", beam_size=5)
    return " ".join(s.text.strip() for s in segments).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--gender", choices=["male", "female"], required=True)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--description", default="")
    ap.add_argument("--bank", default=None, help="voice bank dir (default: config voices.bank)")
    args = ap.parse_args()

    if not 3.0 <= args.end - args.start <= 15.0:
        ap.error("clip length must be 3–15 s (F5-TTS reference sweet spot is 5–10 s)")

    bank = bank_dir(Path(args.bank) if args.bank else None)
    bank.mkdir(parents=True, exist_ok=True)
    wav = bank / f"{args.name}.wav"
    subprocess.run([
        FFMPEG_EXE, "-y", "-v", "error",
        "-ss", str(args.start), "-to", str(args.end), "-i", args.source,
        "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(wav),
    ], check=True)

    ref_text = args.ref_text or _transcribe_vi(wav)
    ref_text = normalize_for_tts(ref_text, use_vinorm=False)

    catalog_path = bank / "catalog.yaml"
    data = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {}
    entries = [e for e in (data.get("voices") or []) if e.get("name") != args.name]
    entries.append({
        "name": args.name, "gender": args.gender, "wav": wav.name,
        "ref_text": ref_text, "description": args.description,
    })
    catalog_path.write_text(yaml.safe_dump({"voices": entries}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Added voice '{args.name}' → {wav}\n  ref_text: {ref_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`config/default.yaml` thêm khối (sau `tts:`):

```yaml
# ── 3b. Voice bank (fixed voices for zero-shot local TTS, e.g. f5vi) ──
voices:
  bank: "assets/voices/vi"   # directory with catalog.yaml + reference WAVs
  default_male: ""           # voice name; empty = first male in catalog
  default_female: ""         # voice name; empty = first female in catalog
```

- [ ] **Step 6: Chạy test**

Run: `uv run python -m pytest -q`
Expected: `54 passed`.

- [ ] **Step 7: Commit**

```bash
git add pipeline/devices.py pipeline/voices.py assets/voices/vi scripts/make_ref_clip.py config/default.yaml tests/test_devices.py tests/test_voices.py
git commit -m "feat(voices): fixed Vietnamese voice bank, device helpers and ref-clip tool"
```

---

### Task 6: Backend TTS `f5vi` + `tts.make_synthesizer()`

**Files:**
- Create: `pipeline/tts_f5vi.py`
- Modify: `pipeline/tts.py:25-38, 56-73` và thêm `make_synthesizer`
- Modify: `config/default.yaml` (khối `f5vi`)
- Modify: `pyproject.toml` (extras)
- Test: `tests/test_tts_f5vi.py`

**Interfaces:**
- Consumes: `voices.get_voice(name) -> Voice`, `vi_text.normalize_for_tts`, `devices.pick_device`, `tts_supertonic._append_silence(path, ms)`.
- Produces (module `tts_f5vi`): `get_shared_tts() -> F5ViEngine`, `F5ViEngine.synthesize(text, ref_wav: str, ref_text: str, speed: float, fix_duration: float | None) -> tuple[np.ndarray, int]`, `synthesize_segment(text, voice, output_path, client, language="vi", speed=None, emotion=None) -> str`, `synthesize_segments(segments, voice, output_dir, client, language, voice_map, tracker, speed, emotion) -> list[str]`, `native_voices_for`, `all_voices`, `voice_descriptions`, `unload()`.
- Produces (module `tts`): `make_synthesizer(language: str, emotion: str | None = None) -> Callable[[str, str, str, float], str]` — `synth(text, voice, out_path, speed) -> out_path`, dùng bởi fitter (Task 9).
- Config `f5vi`: `model_dir, ckpt_file, vocab_file, model_name, vocoder, device, nfe_step, cfg_strength, seed, use_vinorm, loanwords`.

- [ ] **Step 1: Viết test thất bại**

`tests/test_tts_f5vi.py`:

```python
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from pipeline import config as pipeline_config
from pipeline import tts, tts_f5vi


class FakeEngine:
    """Records calls; returns `seconds` of 24 kHz sine per call."""

    def __init__(self, seconds=1.0):
        self.seconds = seconds
        self.calls: list[dict] = []

    def synthesize(self, text, ref_wav, ref_text, speed=1.0, fix_duration=None):
        self.calls.append(dict(text=text, ref_wav=ref_wav, ref_text=ref_text, speed=speed, fix_duration=fix_duration))
        sr = 24000
        n = int(sr * self.seconds / speed)
        return (0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / sr)).astype(np.float32), sr


@pytest.fixture
def bank(tmp_path, monkeypatch):
    sr = 24000
    sf.write(tmp_path / "nam-1.wav", np.zeros(sr * 5, dtype=np.float32), sr)
    (tmp_path / "catalog.yaml").write_text(yaml.safe_dump({"voices": [
        {"name": "nam-1", "gender": "male", "wav": "nam-1.wav", "ref_text": "xin chào các bạn"}]}), encoding="utf-8")
    cfg = pipeline_config.load()
    monkeypatch.setitem(cfg["voices"], "bank", str(tmp_path))
    monkeypatch.setitem(cfg["tts"], "sentence_tail_silence_ms", 250)
    monkeypatch.setitem(cfg["tts"], "tail_silence_ms", 120)
    monkeypatch.setitem(cfg["f5vi"], "loanwords", {"GPU": "gi pi u"})
    monkeypatch.setitem(cfg["f5vi"], "use_vinorm", False)
    return tmp_path


def test_synthesize_segment_writes_44k_mono_with_sentence_tail(bank, tmp_path):
    engine = FakeEngine(seconds=1.0)
    out = tts_f5vi.synthesize_segment("Xin chào.", "nam-1", str(tmp_path / "o.wav"), engine, language="vi")
    info = sf.info(out)
    assert info.samplerate == 44100 and info.channels == 1
    assert abs(info.duration - 1.25) < 0.03


def test_text_is_normalized_and_ref_from_bank(bank, tmp_path):
    engine = FakeEngine()
    tts_f5vi.synthesize_segment("Có 2 GPU", "nam-1", str(tmp_path / "o.wav"), engine, language="vi")
    call = engine.calls[0]
    assert call["text"] == "có hai gi pi u"
    assert call["ref_text"] == "xin chào các bạn"
    assert Path(call["ref_wav"]).name == "nam-1.wav"


def test_speed_is_forwarded_and_shortens_audio(bank, tmp_path):
    engine = FakeEngine(seconds=2.0)
    out = tts_f5vi.synthesize_segment("Một câu", "nam-1", str(tmp_path / "o.wav"), engine, language="vi", speed=1.15)
    assert engine.calls[0]["speed"] == pytest.approx(1.15)
    assert sf.info(out).duration < 2.0


def test_make_synthesizer_uses_f5vi_backend(bank, tmp_path, monkeypatch):
    cfg = pipeline_config.get()
    monkeypatch.setitem(cfg["models"], "tts", {"provider": "f5vi", "model": "f5-tts-vi"})
    engine = FakeEngine(seconds=0.5)
    monkeypatch.setattr(tts_f5vi, "get_shared_tts", lambda: engine)
    synth = tts.make_synthesizer(language="vi")
    out = synth("Xin chào", "nam-1", str(tmp_path / "s.wav"), 1.0)
    assert Path(out).exists() and len(engine.calls) == 1


def test_native_voices_come_from_bank(bank):
    assert tts_f5vi.native_voices_for("vi") == ["nam-1", "nam-1"]
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `uv run python -m pytest tests/test_tts_f5vi.py -v`
Expected: FAIL — `ModuleNotFoundError: pipeline.tts_f5vi`.

- [ ] **Step 3: Viết `pipeline/tts_f5vi.py`**

```python
"""Local Vietnamese TTS backend: F5-TTS (PyTorch) + hynt/F5-TTS-Vietnamese-ViVoice.

Runs on CUDA / Apple MPS / CPU. Model files live in `f5vi.model_dir`
(default ~/.cache/violin/f5vi): `model_last.pt` + `vocab.txt` (the HF repo's
`config.json` IS the vocab — rename it). License CC-BY-NC-SA-4.0: personal /
research use only.

Fixed voices come from the voice bank (pipeline.voices): each voice is a
reference clip + transcript; F5-TTS clones that timbre for every segment.
Contract mirrors pipeline/tts_supertonic.py so pipeline/tts.py can dispatch.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from . import config as _conf
from . import voices as _voices
from .costs import CostTracker
from .devices import pick_device
from .ffmpeg_utils import FFMPEG_EXE
from .transcriber import Segment
from .tts_supertonic import _append_silence
from .vi_text import normalize_for_tts

_LOCK = threading.Lock()
_ENGINE: "F5ViEngine | None" = None


def _fcfg() -> dict[str, Any]:
    return _conf.get().get("f5vi", {})


class F5ViEngine:
    """Thin wrapper over f5_tts.infer.utils_infer with per-voice reference caching."""

    def __init__(self, cfg: dict[str, Any]):
        from importlib.resources import files
        from omegaconf import OmegaConf
        import torch
        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from f5_tts.model import DiT

        self.cfg = cfg
        self.device = pick_device(cfg.get("device", "auto"))
        model_dir = Path(os.path.expanduser(cfg.get("model_dir", "~/.cache/violin/f5vi")))
        ckpt = model_dir / cfg.get("ckpt_file", "model_last.pt")
        vocab = model_dir / cfg.get("vocab_file", "vocab.txt")
        if not ckpt.exists() or not vocab.exists():
            raise RuntimeError(
                f"F5-TTS-vi files missing in {model_dir}: need {ckpt.name} and {vocab.name}.\n"
                "Download from https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice "
                "(model_last.pt, and config.json renamed to vocab.txt)."
            )
        model_name = cfg.get("model_name", "F5TTS_Base")
        arch_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model_name}.yaml")))
        self.mel_spec_type = cfg.get("vocoder", "vocos")
        print(f"      [f5vi] loading {model_name} on {self.device} from {model_dir}…")
        self.model = load_model(
            DiT, arch_cfg.model.arch, str(ckpt),
            mel_spec_type=self.mel_spec_type, vocab_file=str(vocab), device=self.device,
        )
        self.vocoder = load_vocoder(vocoder_name=self.mel_spec_type, is_local=False, device=self.device)
        self._torch = torch
        self._ref_cache: dict[str, tuple[str, str]] = {}

    def _ref(self, ref_wav: str, ref_text: str) -> tuple[str, str]:
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text
        key = f"{ref_wav}|{ref_text}"
        if key not in self._ref_cache:
            self._ref_cache[key] = preprocess_ref_audio_text(ref_wav, ref_text)
        return self._ref_cache[key]

    def synthesize(
        self, text: str, ref_wav: str, ref_text: str,
        speed: float = 1.0, fix_duration: float | None = None,
    ) -> tuple[np.ndarray, int]:
        from f5_tts.infer.utils_infer import infer_process
        seed = int(self.cfg.get("seed", -1))
        if seed >= 0:
            self._torch.manual_seed(seed)
        audio_ref, text_ref = self._ref(ref_wav, ref_text)
        wav, sr, _spec = infer_process(
            audio_ref, text_ref, text, self.model, self.vocoder,
            mel_spec_type=self.mel_spec_type,
            speed=float(speed),
            nfe_step=int(self.cfg.get("nfe_step", 32)),
            cfg_strength=float(self.cfg.get("cfg_strength", 2.0)),
            fix_duration=fix_duration,
            device=self.device,
        )
        return np.asarray(wav, dtype=np.float32), int(sr)


def get_shared_tts() -> F5ViEngine:
    global _ENGINE
    if _ENGINE is None:
        with _LOCK:
            if _ENGINE is None:
                _ENGINE = F5ViEngine(_fcfg())
    return _ENGINE


def unload() -> None:
    global _ENGINE
    with _LOCK:
        _ENGINE = None
    from .devices import free_memory
    free_memory()


# ── voice catalog (delegates to the voice bank) ─────────────

def native_voices_for(language_code: str) -> list[str]:
    return _voices.native_voices_for(language_code)


def all_voices() -> dict[str, list[str]]:
    return _voices.all_voices()


def voice_descriptions() -> dict[str, str]:
    return _voices.voice_descriptions()


# ── synthesis ───────────────────────────────────────────────

def _write_44k_mono(wav: np.ndarray, sr: int, output_path: str) -> None:
    """Write float audio to disk, resampling to 44.1 kHz mono PCM16 via ffmpeg."""
    tmp = output_path + ".raw.wav"
    sf.write(tmp, wav, sr, subtype="PCM_16")
    subprocess.run(
        [FFMPEG_EXE, "-y", "-v", "error", "-i", tmp, "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", output_path],
        check=True, capture_output=True,
    )
    Path(tmp).unlink(missing_ok=True)


def synthesize_segment(
    text: str,
    voice: str,
    output_path: str,
    client: Any,
    language: str = "vi",
    speed: float | None = None,
    emotion: str | None = None,
) -> str:
    """Synthesize one segment with the fixed voice `voice`; write 44.1 kHz mono WAV."""
    _ = emotion  # F5-TTS has no emotion control
    engine = client if client is not None else get_shared_tts()
    fcfg = _fcfg()
    v = _voices.get_voice(voice)
    gen_text = normalize_for_tts(
        text,
        use_vinorm=None if fcfg.get("use_vinorm", "auto") == "auto" else bool(fcfg.get("use_vinorm")),
        loanwords=fcfg.get("loanwords") or {},
    ) if language.lower().startswith("vi") else text
    spd = max(1.0, min(2.0, float(speed if speed is not None else fcfg.get("speed", 1.0))))

    with _LOCK:
        wav, sr = engine.synthesize(gen_text, str(v.ref_wav), v.ref_text, speed=spd)
    _write_44k_mono(wav, sr, output_path)

    tcfg = _conf.get().get("tts", {})
    if text.rstrip().endswith((".", "!", "?", "。", "！", "？")):
        tail_ms = tcfg.get("sentence_tail_silence_ms", tcfg.get("tail_silence_ms", 0))
    else:
        tail_ms = tcfg.get("tail_silence_ms", 0)
    _append_silence(output_path, tail_ms)
    return output_path


def synthesize_segments(
    segments: list[Segment],
    voice: str,
    output_dir: str,
    client: Any,
    language: str = "vi",
    voice_map: dict[str, str] | None = None,
    tracker: CostTracker | None = None,
    speed: float | None = None,
    emotion: str | None = None,
) -> list[str]:
    """Serial synthesis (F5 saturates the accelerator; no thread pool)."""
    vm = voice_map or {}
    paths: list[str] = []
    for i, seg in enumerate(segments):
        path = str(Path(output_dir) / f"seg_{seg.id:05d}.wav")
        synthesize_segment(seg.text, vm.get(seg.speaker, voice), path, client, language, speed, emotion)
        if tracker:
            tracker.add_tts_usage(len(seg.text))
        paths.append(path)
        if (i + 1) % 10 == 0 or i + 1 == len(segments):
            print(f"      TTS progress: {i + 1}/{len(segments)} segments done")
    return paths
```

Lưu ý spike M0(a): nếu `f5_tts` phiên bản cài đặt đổi tên tham số (`load_model(..., vocab_file=)`, `infer_process(..., fix_duration=)`), sửa cho khớp — kiểm tra bằng `uv run python -c "import inspect, f5_tts.infer.utils_infer as u; print(inspect.signature(u.load_model)); print(inspect.signature(u.infer_process))"`.

- [ ] **Step 4: Nối vào `pipeline/tts.py`**

Trong `_backend`: thêm trước nhánh `else`:

```python
    elif p == "f5vi":
        from . import tts_f5vi as _imp
```

Trong `_make_client`: thêm đầu hàm:

```python
    if provider == "f5vi":
        from . import tts_f5vi as _f5
        return _f5.get_shared_tts()
```

Thêm cuối file:

```python
def make_synthesizer(language: str = "en", emotion: str | None = None):
    """Return `synth(text, voice, out_path, speed) -> out_path` bound to the active backend.

    Used by pipeline.fitter, which needs per-segment speed control and calls
    TTS one segment at a time.
    """
    provider = get_tts_provider()
    backend = _backend(provider)
    client = _make_client(provider)

    def _synth(text: str, voice: str, out_path: str, speed: float = 1.0) -> str:
        return backend.synthesize_segment(text, voice, out_path, client, language, speed, emotion)

    return _synth
```

- [ ] **Step 5: Config + extras**

`config/default.yaml` thêm khối sau `voices:`:

```yaml
# ── 3c. F5-TTS Vietnamese (only used when models.tts.provider = f5vi) ──
f5vi:
  model_dir: "~/.cache/violin/f5vi"   # holds model_last.pt + vocab.txt (HF: hynt/F5-TTS-Vietnamese-ViVoice)
  ckpt_file: model_last.pt
  vocab_file: vocab.txt
  model_name: F5TTS_Base              # arch config name inside the f5_tts package
  vocoder: vocos
  device: auto                        # auto | cuda | mps | cpu
  nfe_step: 32                        # 16 = faster, 32 = default quality
  cfg_strength: 2.0
  speed: 1.0                          # natural; fitter raises per segment up to fit.max_tts_speed
  seed: -1                            # >= 0 for reproducible output
  use_vinorm: auto                    # auto (if installed) | true | false
  loanwords: {}                       # e.g. {"GPU": "gi pi u", "API": "a pi ai"}
```

`pyproject.toml` optional-dependencies:

```toml
[project.optional-dependencies]
# Fully-local stack. Mac: `uv sync --extra local-mac`; NVIDIA: `uv sync --extra local-gpu`.
local-mac = [
    "faster-whisper>=1.1.0",
    "supertonic>=0.1.0",
    "f5-tts>=1.1.0",
    "vinorm>=2.0.7",
    "librosa>=0.10",
]
local-gpu = [
    "faster-whisper>=1.1.0",
    "f5-tts>=1.1.0",
    "vinorm>=2.0.7",
    "librosa>=0.10",
]
local = [   # backwards-compatible alias of local-mac
    "faster-whisper>=1.1.0",
    "supertonic>=0.1.0",
    "f5-tts>=1.1.0",
    "vinorm>=2.0.7",
    "librosa>=0.10",
]
```

Chạy `uv sync --extra local-mac` (tải torch ~ vài phút). Nếu `vinorm` không cài được trên arm64 (spike M0(e)), bỏ khỏi extras và ghi vào README; `vi_text` đã có fallback.

- [ ] **Step 6: Chạy test**

Run: `uv run python -m pytest -q`
Expected: `59 passed`.

- [ ] **Step 7: Commit**

```bash
git add pipeline/tts_f5vi.py pipeline/tts.py config/default.yaml pyproject.toml uv.lock tests/test_tts_f5vi.py
git commit -m "feat(tts): local F5-TTS Vietnamese backend with voice bank and make_synthesizer"
```

---

### Task 7: Prompt có ngân sách thời gian + `shorten_segment()`

**Files:**
- Modify: `prompts/translate.yaml` (`batch_user`, `batch_user_styled`, thêm `shorten_system`, `shorten_user`)
- Modify: `pipeline/translator.py` (`_try_batch`, `_translate_batch`, `translate_segments` nhận `budgets`; thêm `shorten_segment`)
- Test: `tests/test_translator_budget.py`

**Interfaces:**
- Produces: `translate_segments(segments, target_language, client, source_language="auto-detect", tracker=None, style_directives="", style_temperature=None, budgets: list[tuple[float, int]] | None = None)` — `budgets[i] = (seconds, max_syllables)`; `shorten_segment(source_text: str, current_text: str, budget_syllables: int, budget_seconds: float, target_language: str, client, tracker=None) -> str` (trả `current_text` nếu LLM lỗi).
- Consumes: `_response_format`, `_is_local_provider` (Task 3), `SINGLE_SCHEMA`.

- [ ] **Step 1: Viết test thất bại**

`tests/test_translator_budget.py`:

```python
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
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `uv run python -m pytest tests/test_translator_budget.py -v`
Expected: FAIL — `TypeError: _try_batch() got an unexpected keyword argument 'budgets'`, `AttributeError: shorten_segment`.

- [ ] **Step 3: Sửa prompt**

Trong `prompts/translate.yaml`, ở `batch_user` thay đoạn từ `CRITICAL — keep translations SHORT…` đến `…It is better to lose minor nuance than to overrun.` bằng:

```yaml
  CRITICAL — keep translations SHORT so they fit the original speaking time:
  - The {target_language} version MUST be SHORTER than the source — not equal, SHORTER.
  - Vietnamese inflates vs. English; counteract aggressively. Paraphrase, don't transliterate:
    · Drop filler ("very", "really", "actually", "basically", "well", "you know", "I mean").
    · Drop redundant clauses ("which is to say", "in other words").
    · Use shortest natural phrasing — single-word verbs, pronouns over re-naming.
    · Collapse "in order to" → "để"; "the way in which we can" → "cách"; "be able to" → "có thể".
  - Target ~0.85× the word count of the source. It is better to lose minor nuance than to overrun.
  {budget_block}
```

Thêm `{budget_block}` tương tự vào `batch_user_styled` ngay sau dòng `- Lose minor nuance to preserve timing.`. Thêm cuối file:

```yaml
## Length budget block (inserted into batch prompts when the fitter supplies per-segment budgets)

budget_block: |
  SYLLABLE BUDGET (hard constraint): each segment below is annotated "(X.Xs, ≤N syllables)" —
  the seconds of speech available and the maximum number of Vietnamese syllables that fit.
  Vietnamese words are monosyllabic, so count words. Never exceed N; aim for 80–90% of N.

## Shorten one over-budget segment (fitter phase A)

shorten_system: |
  You are an expert {target_language} dubbing editor. You shorten a {target_language} translation so it can be
  spoken within a strict time budget while preserving the core meaning of the original sentence.
  Return a JSON object with a single key "translation" holding the shortened {target_language} text.

shorten_user: |
  Original ({source_language}): {source_text}
  Current {target_language} (too long, {current_syllables} syllables): {current_text}
  Budget: at most {budget_syllables} syllables (about {budget_seconds} seconds of speech).

  Rules:
  - Keep the meaning; drop filler, repetition and redundant clauses first.
  - Prefer short native phrasing; keep product names and standard acronyms.
  - Output must be 100% {target_language}, one sentence, no explanations.
  - Return JSON: {{"translation": "<shortened text>"}}
```

- [ ] **Step 4: Sửa `pipeline/translator.py`**

`_try_batch` nhận `budgets: list[tuple[float, int]] | None = None`; thay cách dựng `numbered`:

```python
    if budgets:
        numbered = "\n".join(
            f"[{i}] ({sec:.1f}s, ≤{syl} syllables): {json.dumps(t, ensure_ascii=False)}"
            for i, (t, (sec, syl)) in enumerate(zip(texts, budgets))
        )
        budget_block = _prompts.load("translate", "budget_block")
    else:
        numbered = "\n".join(f"[{i}]: {json.dumps(t, ensure_ascii=False)}" for i, t in enumerate(texts))
        budget_block = ""
```

và thêm `budget_block=budget_block` vào `fmt`. `_translate_batch` nhận `budgets` và cắt theo `[:mid]` / `[mid:]` khi chia đôi (giữ `None` nếu không có). `translate_segments` nhận `budgets` và truyền `budgets[i:i+batch_size] if budgets else None`.

Thêm:

```python
def shorten_segment(
    source_text: str,
    current_text: str,
    budget_syllables: int,
    budget_seconds: float,
    target_language: str,
    client: Any,
    tracker: CostTracker | None = None,
    source_language: str = "English",
) -> str:
    """Ask the LLM for a shorter translation that fits `budget_syllables`.

    Returns `current_text` unchanged when the model fails, so the fitter can
    always continue (speed-up + merger will absorb the overrun instead).
    """
    from .vi_text import count_syllables

    cfg = _conf.get()
    fmt = dict(
        source_language=source_language,
        target_language=target_language,
        source_text=source_text,
        current_text=current_text,
        current_syllables=count_syllables(current_text),
        budget_syllables=budget_syllables,
        budget_seconds=f"{budget_seconds:.1f}",
    )
    system_msg = _prompts.load("translate", "shorten_system", **fmt)
    user_msg = _prompts.load("translate", "shorten_user", **fmt)
    if _is_local_provider(cfg):
        system_msg = "/no_think\n" + system_msg
    try:
        response = client.chat.completions.create(
            model=get_translation_model(cfg),
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.2,
            response_format=_response_format("shortened_translation", SINGLE_SCHEMA),
            **_together_extra(),
        )
        if tracker and getattr(response, "usage", None):
            tracker.add_llm_usage(response.usage.prompt_tokens or 0, response.usage.completion_tokens or 0)
        text = json.loads(response.choices[0].message.content.strip())["translation"].strip()
        return text or current_text
    except Exception as exc:  # JSON errors, API errors — never break the pipeline
        print(f"        ⚠ shorten failed ({exc}); keeping current text")
        return current_text
```

- [ ] **Step 5: Chạy test**

Run: `uv run python -m pytest -q`
Expected: `63 passed`.

- [ ] **Step 6: Commit**

```bash
git add prompts/translate.yaml pipeline/translator.py tests/test_translator_budget.py
git commit -m "feat(translator): per-segment syllable budgets in batch prompt and shorten_segment"
```

---

### Task 8: `merge_continuous_segments(min_duration=)` gộp đơn vị quá ngắn

**Files:**
- Modify: `pipeline/transcriber.py:109-157`
- Modify: `config/default.yaml` khối `merge` (thêm `min_duration: 0.0`)
- Test: `tests/test_transcriber_min_duration.py`

**Interfaces:**
- Produces: `merge_continuous_segments(segments, max_gap=None, max_duration=None, min_duration=None)`; `min_duration` mặc định từ `merge.min_duration` (0 = tắt).

- [ ] **Step 1: Viết test thất bại**

`tests/test_transcriber_min_duration.py`:

```python
from pipeline import config as pipeline_config
from pipeline.transcriber import Segment, merge_continuous_segments


def setup_module(module):
    pipeline_config.load()


def _s(i, a, b, text, spk="SPEAKER_00"):
    return Segment(id=i, start=a, end=b, text=text, speaker=spk)


def test_short_segment_is_absorbed_into_previous_same_speaker():
    segs = [_s(0, 0.0, 3.0, "First sentence."), _s(1, 3.2, 4.0, "Yes."), _s(2, 4.5, 8.0, "Third one.")]
    out = merge_continuous_segments(segs, min_duration=2.5)
    assert [s.text for s in out] == ["First sentence. Yes.", "Third one."]
    assert out[0].start == 0.0 and out[0].end == 4.0


def test_short_segment_not_merged_across_speakers():
    segs = [_s(0, 0.0, 3.0, "Hello.", "SPEAKER_00"), _s(1, 3.2, 4.0, "Hi.", "SPEAKER_01")]
    out = merge_continuous_segments(segs, min_duration=2.5)
    assert len(out) == 2


def test_min_duration_zero_keeps_behaviour():
    segs = [_s(0, 0.0, 3.0, "Hello."), _s(1, 3.2, 4.0, "Hi.")]
    assert len(merge_continuous_segments(segs, min_duration=0.0)) == 2


def test_short_first_segment_absorbs_next():
    segs = [_s(0, 0.0, 0.8, "Okay."), _s(1, 1.0, 4.0, "So today we start.")]
    out = merge_continuous_segments(segs, min_duration=2.5)
    assert len(out) == 1 and out[0].text == "Okay. So today we start."
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `uv run python -m pytest tests/test_transcriber_min_duration.py -v`
Expected: FAIL — `TypeError: unexpected keyword argument 'min_duration'`.

- [ ] **Step 3: Sửa `merge_continuous_segments`**

Chữ ký và đọc config:

```python
def merge_continuous_segments(
    segments: list["Segment"],
    max_gap: float | None = None,
    max_duration: float | None = None,
    min_duration: float | None = None,
) -> list["Segment"]:
```

```python
    if min_duration is None:
        min_duration = float(cfg.get("min_duration", 0.0) or 0.0)
```

Ngay trước vòng `for i, seg in enumerate(merged): seg.id = i`:

```python
    merged = _absorb_short_segments(merged, min_duration, max_gap)
```

Thêm hàm mới phía trên `merge_continuous_segments`:

```python
def _absorb_short_segments(segments: list["Segment"], min_duration: float, max_gap: float) -> list["Segment"]:
    """Merge segments shorter than *min_duration* into a same-speaker neighbour.

    Sub-2.5 s units make TTS choppy and give the duration fitter no room to
    work; a short unit is glued to the previous segment when the gap allows,
    otherwise the following one absorbs it.
    """
    if min_duration <= 0 or len(segments) < 2:
        return segments

    def _join(a: "Segment", b: "Segment") -> "Segment":
        return Segment(id=a.id, start=a.start, end=b.end, text=(a.text + " " + b.text).strip(),
                       speaker=a.speaker, source_text=(a.source_text + " " + b.source_text).strip())

    out: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = out[-1]
        close = seg.speaker == prev.speaker and (seg.start - prev.end) <= max_gap
        short_cur = (seg.end - seg.start) < min_duration
        short_prev = (prev.end - prev.start) < min_duration
        if close and (short_cur or short_prev):
            out[-1] = _join(prev, seg)
        else:
            out.append(seg)
    return out
```

`config/default.yaml` khối `merge` thêm:

```yaml
  min_duration: 0.0      # seconds — glue same-speaker segments shorter than this to a neighbour (0 = off; local presets use 2.5)
```

- [ ] **Step 4: Chạy test**

Run: `uv run python -m pytest -q`
Expected: `67 passed`.

- [ ] **Step 5: Commit**

```bash
git add pipeline/transcriber.py config/default.yaml tests/test_transcriber_min_duration.py
git commit -m "feat(transcriber): min_duration merge for too-short segments"
```

---

### Task 9: `pipeline/fitter.py` — vòng lặp fit-duration

**Files:**
- Create: `pipeline/fitter.py`
- Modify: `config/default.yaml` (khối `fit`)
- Test: `tests/test_fitter.py`

**Interfaces:**
- Consumes: `vi_text.count_syllables`, `Segment`, `synth(text, voice, out_path, speed) -> path` (Task 6), `shorten_fn(source_text, current_text, budget_syllables, budget_seconds) -> str` (bọc `translator.shorten_segment`, Task 7).
- Produces:
  - `DubUnit` dataclass (fields: `seg_id, speaker, voice, source_text, text, start, end, slot_end, syllables, est_s, tts_path, tts_dur, tts_speed, strategy, rounds, over_s`; property `budget_s`).
  - `compute_slots(segments, total_duration, max_borrow_s, margin_s) -> list[float]` (slot_end mỗi segment).
  - `budgets_for(segments, slots, sec_per_syllable) -> list[tuple[float, int]]` (cho `translate_segments`).
  - `build_units(segments, slots, voice_map, default_voice) -> list[DubUnit]`.
  - `fit_text(units, shorten_fn, fcfg) -> None`, `fit_audio(units, synth, out_dir, fcfg) -> None`.
  - `apply_units(units, segments) -> tuple[list[Segment], list[str]]`, `save_units(units, path) -> None`, `wav_duration(path) -> float`.
- Config `fit`: `enabled, sec_per_syllable, overrun_tolerance, max_shorten_rounds, max_tts_speed, max_pause_borrow_s, margin_s`.

- [ ] **Step 1: Viết test thất bại**

`tests/test_fitter.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline import fitter
from pipeline.transcriber import Segment

FCFG = dict(sec_per_syllable=0.2, overrun_tolerance=1.3, max_shorten_rounds=2,
            max_tts_speed=1.15, max_pause_borrow_s=0.6, margin_s=0.05)


def _segs():
    return [
        Segment(id=0, start=0.0, end=2.0, text="một hai ba bốn năm", source_text="one two"),
        Segment(id=1, start=3.0, end=5.0, text="sáu bảy", source_text="six seven"),
        Segment(id=2, start=5.2, end=7.0, text="tám", source_text="eight"),
    ]


def test_compute_slots_borrows_pause_up_to_next_onset():
    slots = fitter.compute_slots(_segs(), total_duration=10.0, max_borrow_s=0.6, margin_s=0.05)
    assert slots == pytest.approx([2.6, 5.15, 7.6])


def test_budgets_for_uses_slot_and_syllable_rate():
    segs = _segs()
    slots = fitter.compute_slots(segs, 10.0, 0.6, 0.05)
    budgets = fitter.budgets_for(segs, slots, sec_per_syllable=0.2)
    assert budgets[0] == (pytest.approx(2.6), 13)


def test_fit_text_shortens_until_within_tolerance():
    segs = _segs()
    slots = [1.0, 5.15, 7.6]                      # unit 0 gets a 1.0 s budget (5 syll ok, tol → 6.5)
    units = fitter.build_units(segs, slots, {}, "nam-1")
    units[0].text = "một hai ba bốn năm sáu bảy tám chín mười"   # 10 syll → est 2.0 s > 1.3
    calls = []

    def shorten(src, cur, syl, sec):
        calls.append((src, cur, syl, sec))
        return "một hai ba bốn"                    # 4 syll → est 0.8 s

    fitter.fit_text(units, shorten, FCFG)
    assert calls == [("one two", "một hai ba bốn năm sáu bảy tám chín mười", 5, pytest.approx(1.0))]
    assert units[0].text == "một hai ba bốn" and units[0].rounds == 1 and units[0].strategy == "shortened"
    assert units[1].rounds == 0


def test_fit_text_stops_when_no_progress():
    units = fitter.build_units(_segs(), [1.0, 5.15, 7.6], {}, "nam-1")
    units[0].text = "a b c d e f g h i j"
    fitter.fit_text(units, lambda *a: "a b c d e f g h i j", FCFG)
    assert units[0].rounds == 1


def _fake_synth(tmp_path):
    calls = []

    def synth(text, voice, out_path, speed=1.0):
        calls.append((text, voice, speed))
        dur = 0.2 * len(text.split()) / speed
        sr = 44100
        sf.write(out_path, np.zeros(int(sr * dur), dtype=np.float32), sr)
        return out_path

    return synth, calls


def test_fit_audio_resynths_over_budget_at_capped_speed(tmp_path):
    units = fitter.build_units(_segs(), [1.0, 5.15, 7.6], {}, "nam-1")   # unit0: 5 syll = 1.0 s at 1.0 → fits exactly
    units[0].text = "a b c d e f g h"                                    # 8 syll = 1.6 s > 1.0
    synth, calls = _fake_synth(tmp_path)
    fitter.fit_audio(units, synth, str(tmp_path), FCFG)
    u = units[0]
    assert u.tts_speed == pytest.approx(1.15)
    assert u.tts_dur == pytest.approx(1.6 / 1.15, abs=0.02)
    assert u.strategy == "tts_speed" and u.over_s > 0
    assert [c[2] for c in calls if c[0] == "a b c d e f g h"] == [1.0, 1.15]
    assert units[1].tts_speed == 1.0 and units[1].strategy == "natural"


def test_apply_units_extends_end_to_borrow_pause(tmp_path):
    segs = _segs()
    units = fitter.build_units(segs, [2.6, 5.15, 7.6], {}, "nam-1")
    units[0].tts_dur, units[0].tts_path = 2.4, "a.wav"     # longer than 2.0 → end becomes 2.4
    units[1].tts_dur, units[1].tts_path = 1.0, "b.wav"     # shorter → keep 5.0
    units[2].tts_dur, units[2].tts_path = 9.0, "c.wav"     # way over → capped at slot_end 7.6
    out, paths = fitter.apply_units(units, segs)
    assert [round(s.end, 2) for s in out] == [2.4, 5.0, 7.6]
    assert paths == ["a.wav", "b.wav", "c.wav"]
    assert out[0].source_text == "one two"


def test_save_units_writes_json(tmp_path):
    units = fitter.build_units(_segs(), [2.6, 5.15, 7.6], {}, "nam-1")
    p = tmp_path / "x.fit.units.json"
    fitter.save_units(units, p)
    data = json.loads(p.read_text())
    assert data["count"] == 3 and data["units"][0]["voice"] == "nam-1"
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `uv run python -m pytest tests/test_fitter.py -v`
Expected: FAIL — `ModuleNotFoundError: pipeline.fitter`.

- [ ] **Step 3: Viết `pipeline/fitter.py`**

```python
"""Duration fitter: make each Vietnamese sentence fit its time slot.

Phase A (`fit_text`, LLM only):  estimate seconds from syllables; when the
    estimate exceeds the slot budget by more than `overrun_tolerance`, ask the
    LLM to shorten the translation (≤ `max_shorten_rounds`).
Phase B (`fit_audio`, TTS only): synthesize at natural speed, measure; if the
    clip is still longer than the budget, re-synthesize once with the TTS
    `speed` raised up to `max_tts_speed`.
`apply_units` then extends `Segment.end` to borrow the following pause
(bounded by `slot_end`); whatever is still over is absorbed by the merger
(video slow-down ≤ 8 %, atempo ≤ 1.4, hard trim). Speech is never slowed.

Units are persisted as `<output>.fit.units.json` for inspection.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import soundfile as sf

from .transcriber import Segment
from .vi_text import count_syllables

ShortenFn = Callable[[str, str, int, float], str]
SynthFn = Callable[[str, str, str, float], str]


@dataclass
class DubUnit:
    seg_id: int
    speaker: str
    voice: str
    source_text: str
    text: str
    start: float
    end: float
    slot_end: float
    syllables: int = 0
    est_s: float = 0.0
    tts_path: str = ""
    tts_dur: float = 0.0
    tts_speed: float = 1.0
    strategy: str = "natural"   # natural | shortened | tts_speed
    rounds: int = 0
    over_s: float = 0.0          # seconds still over budget after phase B (merger absorbs)

    @property
    def budget_s(self) -> float:
        return max(0.0, self.slot_end - self.start)


def wav_duration(path: str) -> float:
    return float(sf.info(path).duration)


def compute_slots(
    segments: list[Segment], total_duration: float, max_borrow_s: float, margin_s: float,
) -> list[float]:
    """slot_end_i = min(end_i + max_borrow, next_start_i - margin, total_duration)."""
    slots: list[float] = []
    for i, seg in enumerate(segments):
        limit = total_duration if i + 1 == len(segments) else segments[i + 1].start - margin_s
        slots.append(max(seg.end, min(seg.end + max_borrow_s, limit)))
    return slots


def budgets_for(
    segments: list[Segment], slots: list[float], sec_per_syllable: float,
) -> list[tuple[float, int]]:
    """(seconds, max_syllables) per segment for the translation prompt."""
    out: list[tuple[float, int]] = []
    for seg, slot_end in zip(segments, slots):
        seconds = max(0.0, slot_end - seg.start)
        out.append((seconds, max(1, int(seconds / sec_per_syllable))))
    return out


def build_units(
    segments: list[Segment], slots: list[float], voice_map: dict[str, str], default_voice: str,
) -> list[DubUnit]:
    return [
        DubUnit(
            seg_id=seg.id, speaker=seg.speaker, voice=voice_map.get(seg.speaker, default_voice),
            source_text=seg.source_text, text=seg.text,
            start=seg.start, end=seg.end, slot_end=slot_end,
        )
        for seg, slot_end in zip(segments, slots)
    ]


def _refresh_estimate(unit: DubUnit, sec_per_syllable: float) -> None:
    unit.syllables = count_syllables(unit.text)
    unit.est_s = unit.syllables * sec_per_syllable


def fit_text(units: list[DubUnit], shorten_fn: ShortenFn, fcfg: dict) -> None:
    sps = float(fcfg.get("sec_per_syllable", 0.21))
    tol = float(fcfg.get("overrun_tolerance", 1.3))
    max_rounds = int(fcfg.get("max_shorten_rounds", 2))
    shortened = 0
    for unit in units:
        _refresh_estimate(unit, sps)
        budget = unit.budget_s
        budget_syll = max(1, int(budget / sps))
        while unit.est_s > budget * tol and unit.rounds < max_rounds:
            new_text = (shorten_fn(unit.source_text, unit.text, budget_syll, budget) or "").strip()
            unit.rounds += 1
            if not new_text or count_syllables(new_text) >= unit.syllables:
                break  # no progress — stop asking
            unit.text = new_text
            unit.strategy = "shortened"
            _refresh_estimate(unit, sps)
        if unit.rounds:
            shortened += 1
    print(f"      [fit] phase A: {shortened}/{len(units)} units shortened")


def fit_audio(units: list[DubUnit], synth: SynthFn, out_dir: str, fcfg: dict) -> None:
    max_speed = float(fcfg.get("max_tts_speed", 1.15))
    os.makedirs(out_dir, exist_ok=True)
    sped = 0
    for i, unit in enumerate(units):
        path = str(Path(out_dir) / f"seg_{unit.seg_id:05d}.wav")
        unit.tts_path = synth(unit.text, unit.voice, path, 1.0)
        unit.tts_dur = wav_duration(unit.tts_path)
        budget = unit.budget_s
        if budget > 0 and unit.tts_dur > budget:
            speed = min(unit.tts_dur / budget, max_speed)
            if speed > 1.02:
                unit.tts_path = synth(unit.text, unit.voice, path, speed)
                unit.tts_dur = wav_duration(unit.tts_path)
                unit.tts_speed = round(speed, 3)
                unit.strategy = "tts_speed"
                sped += 1
        unit.over_s = round(max(0.0, unit.tts_dur - budget), 3)
        if (i + 1) % 10 == 0 or i + 1 == len(units):
            print(f"      [fit] phase B: {i + 1}/{len(units)} synthesized ({sped} re-sped)")


def apply_units(units: list[DubUnit], segments: list[Segment]) -> tuple[list[Segment], list[str]]:
    """Return (segments with borrowed pauses, tts_paths) in the same order."""
    by_id = {u.seg_id: u for u in units}
    out: list[Segment] = []
    paths: list[str] = []
    for seg in segments:
        u = by_id[seg.id]
        new_end = seg.end
        if u.tts_dur > (seg.end - seg.start):
            new_end = min(seg.start + u.tts_dur, u.slot_end)
        out.append(Segment(id=seg.id, start=seg.start, end=new_end, text=u.text,
                           speaker=seg.speaker, source_text=seg.source_text))
        paths.append(u.tts_path)
    return out, paths


def save_units(units: list[DubUnit], path: str | Path) -> None:
    payload = {"count": len(units), "units": [asdict(u) | {"budget_s": round(u.budget_s, 3)} for u in units]}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      [fit] units → {path}")
```

`config/default.yaml` thêm khối trước `# ── 5. Video merge`:

```yaml
# ── 4b. Duration fitter (local dubbing; off by default) ─────
fit:
  enabled: false
  sec_per_syllable: 0.21      # seconds per Vietnamese syllable at TTS speed 1.0 — calibrate per voice (spike M0a)
  overrun_tolerance: 1.30     # ask the LLM to shorten only when est > budget × this
  max_shorten_rounds: 2
  max_tts_speed: 1.15         # F5-TTS `speed` ceiling before handing the residual to the merger
  max_pause_borrow_s: 0.6     # how much of the following pause a sentence may borrow
  margin_s: 0.05              # keep this gap before the next sentence's onset
```

- [ ] **Step 4: Chạy test**

Run: `uv run python -m pytest tests/test_fitter.py -v && uv run python -m pytest -q`
Expected: 8 passed; tổng `75 passed`.

- [ ] **Step 5: Commit**

```bash
git add pipeline/fitter.py config/default.yaml tests/test_fitter.py
git commit -m "feat(fitter): syllable-budget shortening, measured TTS re-speed and pause borrowing"
```

---

### Task 10: Merger `hard_trim` — cắt + fade thay vì freeze khi vẫn tràn

**Files:**
- Modify: `pipeline/merger.py:341-355` (`build_aligned_video`), `:486-500` và `:527-533` (`_build_video_audio_track`, nhánh `tts` và `tts_mixed`)
- Modify: `config/default.yaml` khối `merge_video` (thêm `hard_trim`, `trim_fade_ms`)
- Test: `tests/test_merger_hard_trim.py`

**Interfaces:**
- Produces: khi `merge_video.hard_trim: true`, `build_aligned_video` không tạo freeze (`freeze_extra = 0`) và audio TTS được fade-out `trim_fade_ms` trước khi bị cắt tại `chunk_dur`. Không đổi chữ ký hàm.

- [ ] **Step 1: Viết test thất bại**

`tests/test_merger_hard_trim.py` (bỏ qua nếu thiếu ffmpeg):

```python
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline import config as pipeline_config
from pipeline import merger
from pipeline.transcriber import Segment

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def _video(path: Path, seconds: float = 6.0) -> str:
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=25:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], check=True)
    return str(path)


def _tone(path: Path, seconds: float) -> str:
    sr = 44100
    t = np.arange(int(sr * seconds)) / sr
    sf.write(path, (0.5 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), sr)
    return str(path)


def test_hard_trim_keeps_video_length_and_fades_audio(tmp_path, monkeypatch):
    cfg = pipeline_config.load()
    monkeypatch.setitem(cfg["merge_video"], "speed_clamp_min", 1.0)   # no video stretch
    monkeypatch.setitem(cfg["merge_video"], "speed_clamp_max", 1.0)
    monkeypatch.setitem(cfg["merge_video"], "max_audio_speedup", 1.0)  # no atempo
    monkeypatch.setitem(cfg["merge_video"], "max_freeze_s", 0.0)
    monkeypatch.setitem(cfg["merge_video"], "hard_trim", True)
    monkeypatch.setitem(cfg["merge_video"], "trim_fade_ms", 80)
    monkeypatch.setitem(cfg["merge_video"], "workers", 2)

    video = _video(tmp_path / "in.mp4")
    tts = _tone(tmp_path / "tts.wav", seconds=3.0)          # 1 s longer than the 2 s slot
    segs = [Segment(id=0, start=1.0, end=3.0, text="x")]
    out = str(tmp_path / "out.mp4")
    new_segs = merger.build_aligned_video(video, segs, [tts], 6.0, out)

    assert new_segs[0].end - new_segs[0].start == pytest.approx(2.0, abs=0.05)   # no freeze added
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", out], capture_output=True, text=True).stdout)
    assert dur == pytest.approx(6.0, abs=0.15)
    # last 40 ms before the trim point must be quieter than the middle of the clip
    audio, sr = sf.read(str(tmp_path / "out.wav")) if False else (None, None)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", out, "-ac", "1", "-ar", "44100", str(tmp_path / "out.wav")], check=True)
    audio, sr = sf.read(str(tmp_path / "out.wav"))
    mid = audio[int(sr * 2.0):int(sr * 2.2)]
    tail = audio[int(sr * 2.96):int(sr * 3.0)]
    assert np.abs(tail).mean() < 0.5 * np.abs(mid).mean()
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `uv run python -m pytest tests/test_merger_hard_trim.py -v`
Expected: FAIL — segment mới dài 3.0 s (freeze 1 s được thêm) hoặc video dài 7 s.

- [ ] **Step 3: Sửa `pipeline/merger.py`**

Trong `build_aligned_video`, sau `freeze_extra = max(0.0, effective_tts_dur - target_dur)`:

```python
        if vcfg.get("hard_trim", False):
            freeze_extra = 0.0  # audio is cut (with fade) at chunk_dur instead of freezing frames
        if freeze_extra < 0.001:
            freeze_extra = 0.0
```

Thêm helper cạnh `_atempo_chain`:

```python
def _trim_fade_filter(chunk_dur: float) -> str:
    """afade-out just before chunk_dur when merge_video.hard_trim is on; '' otherwise."""
    vcfg = _conf.get()["merge_video"]
    if not vcfg.get("hard_trim", False):
        return ""
    fade = max(0.01, float(vcfg.get("trim_fade_ms", 80)) / 1000.0)
    start = max(0.0, chunk_dur - fade)
    return f"afade=t=out:st={start:.3f}:d={fade:.3f}"
```

Trong `_build_video_audio_track` nhánh `tts`, thay việc dựng `af_parts`:

```python
            af_parts: list[str] = []
            if audio_speedup > 1.001:
                af_parts.append(f"atempo={audio_speedup}")
            af_parts.append(f"apad=whole_dur={chunk_dur}")
            fade = _trim_fade_filter(chunk_dur)
            if fade:
                af_parts.append(fade)
```

Nhánh `tts_mixed`, sau `tts_filter_parts.append(f"apad=whole_dur={chunk_dur}")`:

```python
            fade = _trim_fade_filter(chunk_dur)
            if fade:
                tts_filter_parts.append(fade)
```

`config/default.yaml` khối `merge_video` thêm:

```yaml
  hard_trim: false        # true: when speech still overruns after stretch+atempo, cut it at the slot (with fade) instead of freezing frames
  trim_fade_ms: 80
```

- [ ] **Step 4: Chạy test**

Run: `uv run python -m pytest -q`
Expected: `76 passed` (test ffmpeg chạy ~5 s).

- [ ] **Step 5: Commit**

```bash
git add pipeline/merger.py config/default.yaml tests/test_merger_hard_trim.py
git commit -m "feat(merger): hard_trim option cuts overrunning speech with a fade instead of freezing frames"
```

---

### Task 11: Nối fitter vào orchestrator, CLI, preset `local_mac.yaml` / `local_gpu.yaml`, resume

**Files:**
- Modify: `pipeline/orchestrator.py:39-56` (`DubOptions`), `:123-191` (luồng chính)
- Modify: `main.py` (`translate_video`, `main`)
- Modify: `config/local_mac.yaml`; Create: `config/local_gpu.yaml`
- Modify: `resume_from_segments.py` (nhận stage `fitted`: bỏ qua dịch và fit, chỉ TTS+merge như `translated`)
- Test: `tests/test_orchestrator_fit.py`

**Interfaces:**
- Consumes: `fitter.*` (Task 9), `tts.make_synthesizer` (Task 6), `translator.shorten_segment`, `translate_segments(..., budgets=)` (Task 7), `merge_continuous_segments(min_duration)` (Task 8).
- Produces: `DubOptions.fit: bool | None = None` (None → `cfg["fit"]["enabled"]`); artifacts `<output>.fitted.segments.json`, `<output>.fit.units.json`; CLI `--no-fit`.

- [ ] **Step 1: Viết test thất bại**

`tests/test_orchestrator_fit.py`:

```python
import tempfile
from pathlib import Path
from unittest.mock import patch

from pipeline import config as pipeline_config
from pipeline.orchestrator import DubOptions, dub_video
from pipeline.transcriber import Segment


def test_fit_path_replaces_synthesize_and_keeps_sentence_units(monkeypatch):
    cfg = pipeline_config.load()
    monkeypatch.setitem(cfg["fit"], "enabled", True)
    monkeypatch.setitem(cfg["models"], "tts", {"provider": "f5vi", "model": "f5-tts-vi"})

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.mp4"
        out.write_bytes(b"video")
        tts = Path(tmp) / "seg.wav"
        tts.write_bytes(b"wav")
        segments = [Segment(id=0, start=0.0, end=2.0, text="Hello. World.")]
        translated = [Segment(id=0, start=0.0, end=2.0, text="Chào. Thế giới.", source_text="Hello. World.")]

        def fake_fit_audio(units, synth, out_dir, fcfg):
            for u in units:
                u.tts_path, u.tts_dur = str(tts), 1.0

        with patch("pipeline.orchestrator.make_translation_client"), \
             patch("pipeline.orchestrator.make_transcription_client"), \
             patch("pipeline.orchestrator.extract_audio", return_value=str(Path(tmp) / "a.wav")), \
             patch("pipeline.orchestrator.get_video_duration", return_value=5.0), \
             patch("pipeline.orchestrator.ensure_video_input", return_value="input.mp4"), \
             patch("pipeline.orchestrator.transcribe", return_value=segments), \
             patch("pipeline.orchestrator.translate_segments", return_value=translated) as tr, \
             patch("pipeline.orchestrator.synthesize_segments") as synth_segments, \
             patch("pipeline.orchestrator.make_synthesizer", return_value=lambda *a: str(tts)), \
             patch("pipeline.orchestrator.fitter.fit_text") as fit_text, \
             patch("pipeline.orchestrator.fitter.fit_audio", side_effect=fake_fit_audio) as fit_audio, \
             patch("pipeline.orchestrator._resolve_voice", return_value="nam-1"), \
             patch("pipeline.orchestrator.prepare_merge", return_value=object()), \
             patch("pipeline.orchestrator.build_gap_chunks"), \
             patch("pipeline.orchestrator.build_aligned_video", side_effect=lambda v, segs, paths, *a, **k: segs):
            result = dub_video("input.mp4", str(out), DubOptions(target_language="Vietnamese", subtitles=False))

        synth_segments.assert_not_called()
        fit_text.assert_called_once()
        fit_audio.assert_called_once()
        assert tr.call_args.kwargs["budgets"] is not None
        # sentence units are NOT re-split after translation on the fit path
        assert [s.text for s in result.aligned_segments] == ["Chào. Thế giới."]
        assert (out.with_suffix(".fit.units.json")).exists()
        assert (out.with_suffix(".fitted.segments.json")).exists()
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `uv run python -m pytest tests/test_orchestrator_fit.py -v`
Expected: FAIL — `AttributeError: pipeline.orchestrator has no attribute 'fitter'` / `make_synthesizer`.

- [ ] **Step 3: Sửa `pipeline/orchestrator.py`**

Import thêm:

```python
from . import fitter
from .translator import shorten_segment, translate_segments
from .tts import make_synthesizer, native_voices_for, synthesize_segments
```

`DubOptions` thêm:

```python
    fit: bool | None = None               # None → config fit.enabled; duration fitter (local dubbing)
```

Thay đoạn từ `lang_code = language_code(...)` đến hết `build_aligned_video(...)` bằng:

```python
        lang_code = language_code(opts.target_language)
        fit_cfg = cfg.get("fit", {})
        fit_enabled = bool(fit_cfg.get("enabled", False)) if opts.fit is None else bool(opts.fit)

        segments = merge_continuous_segments(segments)
        _persist_segments(segments, output_video_path, "transcribed")

        budgets = None
        slots: list[float] = []
        if fit_enabled:
            slots = fitter.compute_slots(
                segments, total_duration,
                float(fit_cfg.get("max_pause_borrow_s", 0.6)), float(fit_cfg.get("margin_s", 0.05)),
            )
            budgets = fitter.budgets_for(segments, slots, float(fit_cfg.get("sec_per_syllable", 0.21)))

        _check_cancel(is_cancelled)
        _emit(on_progress, 3, f"Translating {len(segments)} segments to {opts.target_language} (style: {style.name})…")
        translated = translate_segments(
            segments, opts.target_language, translation_client, opts.source_language,
            tracker=tracker,
            style_directives=style.translation_directives,
            style_temperature=style.temperature,
            budgets=budgets,
        )
        tracker.record_step("Translation (LLM)")
        _persist_segments(translated, output_video_path, "translated")
        if not fit_enabled:
            # Aggressive re-merge → re-split: gives the translator full paragraph
            # context (better quality) while still producing sentence-level units
            # for TTS and subtitles (1-to-1 alignment, readable line lengths).
            translated = merge_continuous_segments(translated, max_duration=float("inf"))
            translated = split_into_sentences(translated)

        _check_cancel(is_cancelled)
        effective_voice = _resolve_voice(opts.voice, lang_code, cfg)
        tts_label = cfg["models"]["tts"]["model"]
        _emit(on_progress, 4, f"Synthesizing TTS with {tts_label} (voice: {effective_voice})…")
        tts_dir = tmp_dir / "tts"
        tts_dir.mkdir()

        mix_volume, original_audio_volume, gap_vol = _voiceover_volumes(opts, cfg)

        if fit_enabled:
            units = fitter.build_units(translated, slots, {}, effective_voice)

            def _shorten(src: str, cur: str, budget_syll: int, budget_s: float) -> str:
                return shorten_segment(src, cur, budget_syll, budget_s, opts.target_language,
                                       translation_client, tracker=tracker)

            fitter.fit_text(units, _shorten, fit_cfg)
            synth = make_synthesizer(language=lang_code, emotion=style.tts_emotion)
            fitter.fit_audio(units, synth, str(tts_dir), fit_cfg)
            translated, tts_paths = fitter.apply_units(units, translated)
            fitter.save_units(units, Path(output_video_path).with_suffix(".fit.units.json"))
            _persist_segments(translated, output_video_path, "fitted")
            tracker.record_step(f"TTS + fit ({tts_label})")
            plan = prepare_merge(
                video_input_path, translated, total_duration,
                preserve_gap_audio=opts.voiceover,
                mix_volume=mix_volume,
                original_audio_volume=original_audio_volume,
                gap_volume=gap_vol,
            )
            build_gap_chunks(plan)
        else:
            plan = prepare_merge(
                video_input_path, translated, total_duration,
                preserve_gap_audio=opts.voiceover,
                mix_volume=mix_volume,
                original_audio_volume=original_audio_volume,
                gap_volume=gap_vol,
            )

            gap_exc: list[Exception] = []

            def _build_gaps():
                try:
                    build_gap_chunks(plan)
                except Exception as e:
                    gap_exc.append(e)

            gap_thread = threading.Thread(target=_build_gaps, daemon=True)
            gap_thread.start()

            tts_paths = synthesize_segments(
                translated, effective_voice, str(tts_dir),
                language=lang_code,
                tracker=tracker,
                speed=style.tts_speed,
                emotion=style.tts_emotion,
                together_api_key=opts.together_api_key,
                elevenlabs_api_key=opts.elevenlabs_api_key,
                openai_api_key=opts.openai_api_key,
            )
            gap_thread.join()
            if gap_exc:
                raise gap_exc[0]
            tracker.record_step(f"TTS ({tts_label})")

        _check_cancel(is_cancelled)
        _emit(on_progress, 5, "Building aligned video…")
        aligned_segments = build_aligned_video(
            video_input_path, translated, tts_paths, total_duration, output_video_path,
            merge_plan=plan,
            original_audio_path=original_audio_path,
        )
```

- [ ] **Step 4: CLI, preset, resume**

`main.py`: `translate_video(...)` thêm tham số `fit: bool | None = None` và `DubOptions(..., fit=fit)`; `main()` thêm:

```python
    parser.add_argument(
        "--no-fit", action="store_true",
        help="Disable the duration fitter even if the config enables it (local dubbing)",
    )
```

và truyền `fit=False if args.no_fit else None`.

`config/local_mac.yaml` viết lại toàn bộ:

```yaml
# Mac M-series preset — FULLY LOCAL (no network after models are downloaded).
#
# STT  : faster-whisper large-v3-turbo (CTranslate2, int8, CPU)
# LLM  : Ollama local — qwen3.5:9b-mlx  (brew install ollama; ollama pull qwen3.5:9b-mlx)
# TTS  : F5-TTS Vietnamese (hynt/F5-TTS-Vietnamese-ViVoice) on MPS, fixed voices from assets/voices/vi
# FIT  : syllable-budget shortening + measured TTS re-speed + pause borrowing
#
# Setup:
#   uv sync --extra local-mac
#   mkdir -p ~/.cache/violin/f5vi && cd ~/.cache/violin/f5vi
#   curl -L -o model_last.pt https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice/resolve/main/model_last.pt
#   curl -L -o vocab.txt     https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice/resolve/main/config.json
#   uv run scripts/make_ref_clip.py --source my_voice.wav --start 0 --end 8 --name nam-1 --gender male
# Run:
#   uv run main.py video.mp4 video_vi.mp4 --language Vietnamese --config config/local_mac.yaml

models:
  transcription:
    provider: faster-whisper
    model: large-v3-turbo
  translation:
    provider: ollama
    model: qwen3.5:9b-mlx
    base_url: http://localhost:11434/v1
  tts:
    provider: f5vi
    model: f5-tts-vi
  chat:
    provider: ollama
    model: qwen3.5:9b-mlx
    base_url: http://localhost:11434/v1

transcription:
  chunk_seconds: 600
  parallel_workers: 1
  local_device: auto
  local_compute_type: auto

translation:
  batch_size: 16
  response_format: json_object          # Ollama: plain JSON mode (strict json_schema can hang)

merge:
  max_subtitle_chars: 0                 # keep whole sentences as TTS units
  min_duration: 2.5                     # glue sub-2.5 s fragments to a neighbour

fit:
  enabled: true
  sec_per_syllable: 0.21                # re-measure with scripts/calibrate (spike M0a) for your voice

f5vi:
  device: auto
  nfe_step: 32

tts:
  workers: 1
  tail_silence_ms: 120
  sentence_tail_silence_ms: 250

merge_video:
  workers: 4
  speed_clamp_min: 0.92                 # video may slow to 92 % to absorb residual overrun
  speed_clamp_max: 1.0
  max_audio_speedup: 1.4
  max_freeze_s: 0.0
  hard_trim: true
  voiceover_volume: 0.02
```

`config/local_gpu.yaml` (mới):

```yaml
# NVIDIA GPU preset — FULLY LOCAL. Same stages as local_mac.yaml on CUDA.
# LLM: Ollama (`ollama pull qwen3.5:27b`) or any OpenAI-compatible server (vLLM) via provider openai_compat.
#   uv sync --extra local-gpu
#   uv run main.py video.mp4 video_vi.mp4 --language Vietnamese --config config/local_gpu.yaml

models:
  transcription:
    provider: faster-whisper
    model: large-v3
  translation:
    provider: ollama                    # or: openai_compat + base_url: http://localhost:8000/v1 (vLLM)
    model: qwen3.5:27b
    base_url: http://localhost:11434/v1
  tts:
    provider: f5vi
    model: f5-tts-vi
  chat:
    provider: ollama
    model: qwen3.5:27b
    base_url: http://localhost:11434/v1

transcription:
  chunk_seconds: 600
  parallel_workers: 1
  local_device: cuda
  local_compute_type: float16

translation:
  batch_size: 32
  response_format: json_object

merge:
  max_subtitle_chars: 0
  min_duration: 2.5

fit:
  enabled: true
  sec_per_syllable: 0.21

f5vi:
  device: cuda
  nfe_step: 32

tts:
  workers: 1
  tail_silence_ms: 120
  sentence_tail_silence_ms: 250

merge_video:
  workers: 16
  speed_clamp_min: 0.92
  speed_clamp_max: 1.0
  max_audio_speedup: 1.4
  max_freeze_s: 0.0
  hard_trim: true
  voiceover_volume: 0.02
```

`resume_from_segments.py`: trong `main()` nơi rẽ nhánh theo `stage`, coi `"fitted"` như `"translated"` (không dịch lại, không re-split; TTS + merge). Cụ thể: nơi có `if stage == "transcribed": … translate …`, phần `else` (translated) đã bỏ qua dịch; thêm điều kiện để với stage `fitted` KHÔNG gọi `merge_continuous_segments(..., inf)` + `split_into_sentences` (đơn vị đã là câu và đã kéo `end`). Cập nhật docstring liệt kê `fitted`.

- [ ] **Step 5: Chạy test**

Run: `uv run python -m pytest -q`
Expected: `77 passed`.

- [ ] **Step 6: Commit**

```bash
git add pipeline/orchestrator.py main.py config/local_mac.yaml config/local_gpu.yaml resume_from_segments.py tests/test_orchestrator_fit.py
git commit -m "feat(pipeline): wire duration fitter into dub_video; fully-local Mac/GPU presets"
```

---

### Task 12: Kiểm thử end-to-end offline + tài liệu

**Files:**
- Modify: `README.md` (mục "Local Vietnamese preset" → hướng dẫn mới)
- Modify: `.claude/skills/video-translator/SKILL.md` (nhắc `--config config/local_mac.yaml`, `--no-fit`)
- Create: `scripts/calibrate_voice.py` (đo `sec_per_syllable` cho một voice)

**Interfaces:**
- Consumes: mọi thứ ở trên.
- Produces: bằng chứng chạy offline; giá trị `fit.sec_per_syllable` đo được ghi vào `config/local_mac.yaml`.

- [ ] **Step 1: Script calibrate**

`scripts/calibrate_voice.py`:

```python
"""Measure seconds-per-syllable of a voice-bank voice at TTS speed 1.0.

    uv run scripts/calibrate_voice.py --voice nam-1 --config config/local_mac.yaml
Prints the value to put in `fit.sec_per_syllable`.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config as pipeline_config  # noqa: E402
from pipeline.fitter import wav_duration  # noqa: E402
from pipeline.tts import make_synthesizer  # noqa: E402
from pipeline.vi_text import count_syllables  # noqa: E402

SENTENCES = [
    "hôm nay chúng ta sẽ tìm hiểu cách mô hình ngôn ngữ lớn học từ dữ liệu văn bản.",
    "trước hết, hãy nhìn vào cấu trúc của một mạng nơ ron đơn giản.",
    "kết quả cho thấy phương pháp mới nhanh hơn khoảng hai lần so với cách cũ.",
    "nếu bạn có câu hỏi, hãy để lại bình luận bên dưới video này.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--config", default="config/local_mac.yaml")
    args = ap.parse_args()
    pipeline_config.load(args.config)
    synth = make_synthesizer(language="vi")
    total_s, total_syl = 0.0, 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, text in enumerate(SENTENCES):
            path = synth(text, args.voice, f"{tmp}/c{i}.wav", 1.0)
            dur = wav_duration(path) - 0.25   # subtract sentence tail silence
            syl = count_syllables(text)
            total_s += dur
            total_syl += syl
            print(f"  {syl:3d} syll  {dur:5.2f}s  → {dur / syl:.3f} s/syll")
    print(f"\nfit.sec_per_syllable: {total_s / total_syl:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Chuẩn bị model + voice + Ollama (một lần)**

```bash
brew list ollama || brew install ollama; brew services start ollama; ollama pull qwen3.5:9b-mlx
mkdir -p ~/.cache/violin/f5vi && cd ~/.cache/violin/f5vi \
  && curl -L -o model_last.pt https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice/resolve/main/model_last.pt \
  && curl -L -o vocab.txt https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice/resolve/main/config.json
cd "<repo>" && uv run scripts/make_ref_clip.py --source <clip giọng Việt của bạn> --start 0 --end 8 --name nam-1 --gender male
uv run scripts/calibrate_voice.py --voice nam-1      # ghi giá trị vào config/local_mac.yaml fit.sec_per_syllable
```

- [ ] **Step 3: Chạy E2E offline**

Tắt Wi-Fi (hoặc `export HF_HUB_OFFLINE=1`), rồi:

```bash
uv run main.py <video tiếng Anh 2–3 phút>.mp4 output/e2e_vi.mp4 --language Vietnamese \
  --config config/local_mac.yaml --subtitle-formats srt,vtt,txt --timings-out output/e2e_timings.json
```

Kiểm tra:
- `output/e2e_vi.mp4` mở được; `ffprobe` độ dài ≤ 1.08 × gốc.
- `output/e2e_vi.fit.units.json`: tỉ lệ unit `strategy == "tts_speed"` < 30 %, `over_s > 0.5` < 10 %.
- Nghe 3 đoạn ngẫu nhiên: giọng bắt đầu đúng lúc người nói gốc bắt đầu (± 0.2 s), không bị cắt giữa câu.
- `output/e2e_timings.json`: thời gian từng bước (ghi vào spec spikes để so với GPU ở M4).

- [ ] **Step 4: Cập nhật README + SKILL**

README mục "Run from source" thay khối "Local Vietnamese workflow" bằng:

```markdown
# Fully-local Vietnamese dubbing (Mac M-series / NVIDIA) — no API keys
uv sync --extra local-mac            # or --extra local-gpu
# one-time: Ollama model, F5-TTS-vi checkpoint, one reference clip — see config/local_mac.yaml header
uv run main.py lecture.mp4 lecture_vi.mp4 --language Vietnamese --config config/local_mac.yaml
```

và một đoạn ngắn mô tả stage fit (`<output>.fit.units.json`, `--no-fit`). Trong `.claude/skills/video-translator/SKILL.md` thêm gợi ý dùng `--config config/local_mac.yaml` cho tiếng Việt offline.

- [ ] **Step 5: Test + commit**

Run: `uv run python -m pytest -q` → `77 passed`.

```bash
git add scripts/calibrate_voice.py README.md .claude/skills/video-translator/SKILL.md config/local_mac.yaml
git commit -m "docs: fully-local Vietnamese dubbing guide and voice calibration script"
```

---

## Self-review (đã chạy khi viết plan)

- **Spec coverage M1:** lỗi có sẵn (Task 1, 8, 11 `max_subtitle_chars: 0`), LLM local (2, 3), vi_text (4), voice bank (5), F5-vi (6), budget + shorten (7), fitter + 2 pha (9), hard_trim (10), orchestrator/CLI/preset/resume (11), E2E + calibrate (12). Chưa thuộc M1 (đúng theo spec): Demucs stems (M2), diarization/gender (M3), WhisperX/vLLM (M4), quality gate (M5) — mỗi cái sẽ có plan riêng.
- **Placeholder scan:** không còn TBD/TODO; mọi bước có code hoặc lệnh cụ thể.
- **Type consistency:** `synth(text, voice, out_path, speed) -> str` dùng nhất quán ở Task 6, 9, 11, 12; `shorten_fn(src, cur, budget_syll, budget_s) -> str` ở Task 7, 9, 11; `budgets: list[tuple[float, int]]` ở Task 7, 9, 11; `Segment.source_text` ở Task 1, 8, 9, 11.
