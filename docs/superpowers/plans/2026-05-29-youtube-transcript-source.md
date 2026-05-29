# YouTube Transcript Source — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Khi input là URL có caption, ưu tiên dùng caption YouTube (manual → auto + LLM thêm dấu câu) làm nguồn script thay vì luôn chạy Whisper; fallback Whisper khi không có caption.

**Architecture:** Module mới `pipeline/captions.py` lấy caption qua yt-dlp, chuẩn hoá thành `list[Segment]`. Caller (CLI/API) gọi nó rồi truyền `segments_override` vào `dub_video`, vốn bỏ qua extract-audio + Whisper khi có sẵn segment. Không đụng `merger.py`.

**Tech Stack:** Python 3.10+, yt-dlp, `urllib`, OpenAI-compatible LLM client (`pipeline/llm_client.py`), `unittest` + `unittest.mock`.

**Spec:** `docs/superpowers/specs/2026-05-29-youtube-transcript-source-design.md`

**Lệnh test (project dùng `unittest`, KHÔNG pytest):**
`uv run python -m unittest tests.test_captions -v`

**Commit:** mỗi commit message kết thúc bằng dòng trailer:
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## File Structure

| File | Trách nhiệm |
|------|-------------|
| `pipeline/captions.py` (mới) | Lấy + chuẩn hoá caption → `Segment[]` |
| `prompts/restore_punctuation.yaml` (mới) | Prompt LLM thêm dấu câu |
| `tests/test_captions.py` (mới) | Unit test cho captions.py |
| `pipeline/orchestrator.py` | `DubOptions.prefer_source_captions` + `dub_video(segments_override=…)` |
| `main.py` | Flag `--no-source-captions` + gọi `fetch_source_captions` |
| `api/worker.py` | `_run_url_job` gọi `fetch_source_captions` |
| `api/models.py`, `api/routes/jobs.py` | field `prefer_source_captions` |
| `api/static/index.html` | checkbox "Ưu tiên phụ đề YouTube" |
| `config/default.yaml` | `transcription.prefer_source_captions: true` |

---

## Task 0: Tạo nhánh + skeleton module

**Files:** Create `pipeline/captions.py`

- [ ] **Step 1: Tạo nhánh làm việc**

```bash
git checkout -b feat/youtube-transcript-source
```

- [ ] **Step 2: Tạo skeleton `pipeline/captions.py`** (imports + dataclasses + hằng số dùng chung cho các task sau)

```python
"""Fetch & normalize source captions (YouTube etc.) into Segment[].

When a video URL already ships captions we prefer them over re-running Whisper:
faster, cheaper, and often more accurate for proper nouns. Manual captions are
used as-is (they carry punctuation); automatic captions are word-level but
unpunctuated, so an LLM restores punctuation and we re-align the punctuated text
back onto the original word timestamps. Any failure falls back to Whisper.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass

from . import config as _conf
from .costs import CostTracker
from .languages import language_code
from .llm_client import get_translation_model, get_translation_provider, make_translation_client
from .transcriber import Segment, _is_sentence_end

import prompts as _prompts

_SENT_END = re.compile(r"[.!?…。！？]+$")
_PUNCT_STRIP = re.compile(r"^[\W_]+|[\W_]+$")  # strip leading/trailing non-word (unicode-aware)

_PUNCT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


@dataclass
class _Word:
    text: str
    start: float
    end: float


@dataclass
class _TrackRef:
    kind: str   # "manual" | "auto"
    lang: str
    url: str
```

- [ ] **Step 3: Verify import OK**

Run: `uv run python -c "import pipeline.captions"`
Expected: không lỗi (exit 0).

- [ ] **Step 4: Commit**

```bash
git add pipeline/captions.py
git commit -m "feat(captions): module skeleton for source-caption ingestion

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 1: Chọn track caption (`_source_lang_code`, `_select_track`)

**Files:** Modify `pipeline/captions.py` · Test `tests/test_captions.py`

- [ ] **Step 1: Viết test thất bại**

```python
import unittest
from pipeline import captions
from pipeline.captions import _TrackRef


def _track(url, ext="json3"):
    return [{"ext": ext, "url": url}]


class SelectTrackTests(unittest.TestCase):
    def test_source_lang_from_explicit_language(self):
        self.assertEqual("en", captions._source_lang_code({}, "English"))

    def test_source_lang_from_info_when_autodetect(self):
        self.assertEqual("en", captions._source_lang_code({"language": "en-US"}, "auto-detect"))

    def test_prefers_manual_exact_over_auto(self):
        info = {
            "subtitles": {"en": _track("MAN")},
            "automatic_captions": {"en": _track("AUTO")},
        }
        t = captions._select_track(info, "en")
        self.assertEqual(("manual", "en", "MAN"), (t.kind, t.lang, t.url))

    def test_uses_auto_asr_when_no_manual(self):
        info = {"automatic_captions": {"en": _track("AUTO"), "vi": _track("TRANS")}}
        t = captions._select_track(info, "en")
        self.assertEqual(("auto", "en", "AUTO"), (t.kind, t.lang, t.url))

    def test_rejects_translated_auto_track(self):
        # only a translated track exists (target=vi), source=en → no ASR original
        info = {"automatic_captions": {"vi": _track("TRANS"), "aa-en": _track("X")}}
        self.assertIsNone(captions._select_track(info, "en"))

    def test_manual_variant_match(self):
        info = {"subtitles": {"en-US": _track("MANUS")}}
        t = captions._select_track(info, "en")
        self.assertEqual(("manual", "en-US", "MANUS"), (t.kind, t.lang, t.url))

    def test_none_when_no_src_and_no_manual(self):
        info = {"automatic_captions": {"en": _track("AUTO")}}
        self.assertIsNone(captions._select_track(info, None))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `uv run python -m unittest tests.test_captions.SelectTrackTests -v`
Expected: FAIL (`AttributeError: module 'pipeline.captions' has no attribute '_source_lang_code'`).

- [ ] **Step 3: Implement (thêm vào `pipeline/captions.py`)**

```python
def _source_lang_code(info: dict, source_language: str) -> str | None:
    if source_language and source_language.lower() != "auto-detect":
        return language_code(source_language)
    lang = info.get("language")
    return lang.split("-")[0] if lang else None


def _json3_url(track_list) -> str | None:
    for e in track_list or []:
        if e.get("ext") == "json3":
            return e.get("url")
    return None


def _select_track(info: dict, src: str | None) -> _TrackRef | None:
    subs = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}

    if src and src in subs:
        u = _json3_url(subs[src])
        if u:
            return _TrackRef("manual", src, u)
    if src:
        for k in sorted(subs):
            if k.split("-")[0] == src:
                u = _json3_url(subs[k])
                if u:
                    return _TrackRef("manual", k, u)
    # auto: ONLY exact source code (ASR original) — avoids translated tracks
    if src and src in auto:
        u = _json3_url(auto[src])
        if u:
            return _TrackRef("auto", src, u)
    if not src and subs:
        k = sorted(subs)[0]
        u = _json3_url(subs[k])
        if u:
            return _TrackRef("manual", k, u)
    return None
```

- [ ] **Step 4: Chạy test — kỳ vọng PASS**

Run: `uv run python -m unittest tests.test_captions.SelectTrackTests -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add pipeline/captions.py tests/test_captions.py
git commit -m "feat(captions): track selection (prefer manual, reject translated auto)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Parse manual caption (`_parse_manual`)

**Files:** Modify `pipeline/captions.py` · `tests/test_captions.py`

- [ ] **Step 1: Viết test thất bại**

```python
class ParseManualTests(unittest.TestCase):
    def test_cue_events_become_segments_with_timestamps(self):
        data = {"events": [
            {"tStartMs": 13240, "dDurationMs": 2560,
             "segs": [{"utf8": "A few years ago,\nI broke in."}]},
            {"tStartMs": 16880, "dDurationMs": 1216,
             "segs": [{"utf8": "I had just driven home,"}]},
            {"tStartMs": 9999, "segs": [{"utf8": "  \n "}]},  # blank → dropped
        ]}
        segs = captions._parse_manual(data)
        self.assertEqual(2, len(segs))
        self.assertEqual("A few years ago, I broke in.", segs[0].text)
        self.assertAlmostEqual(13.24, segs[0].start, places=2)
        self.assertAlmostEqual(15.80, segs[0].end, places=2)
        self.assertEqual([0, 1], [s.id for s in segs])
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `uv run python -m unittest tests.test_captions.ParseManualTests -v`
Expected: FAIL (`has no attribute '_parse_manual'`).

- [ ] **Step 3: Implement**

```python
def _parse_manual(data: dict) -> list[Segment]:
    out: list[Segment] = []
    for ev in data.get("events") or []:
        segs = ev.get("segs")
        if not segs:
            continue
        text = re.sub(r"\s+", " ", "".join(s.get("utf8", "") for s in segs)).strip()
        if not text:
            continue
        start = (ev.get("tStartMs") or 0) / 1000.0
        dur = (ev.get("dDurationMs") or 0) / 1000.0
        out.append(Segment(id=len(out), start=start, end=start + dur, text=text))
    return out
```

- [ ] **Step 4: Chạy test — kỳ vọng PASS**

Run: `uv run python -m unittest tests.test_captions.ParseManualTests -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/captions.py tests/test_captions.py
git commit -m "feat(captions): parse manual cue-level captions

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Parse auto-caption word-level (`_parse_auto_words`)

**Files:** Modify `pipeline/captions.py` · `tests/test_captions.py`

- [ ] **Step 1: Viết test thất bại**

```python
class ParseAutoWordsTests(unittest.TestCase):
    def test_word_level_with_offsets_and_dedup(self):
        data = {"events": [
            {"tStartMs": 1000, "dDurationMs": 2000, "segs": [
                {"utf8": "a", "tOffsetMs": 0},
                {"utf8": " few", "tOffsetMs": 200},
            ]},
            {"tStartMs": 1500, "segs": [{"utf8": "\n"}]},  # noise → dropped
            {"tStartMs": 1180, "dDurationMs": 1500, "segs": [
                {"utf8": "few", "tOffsetMs": 20},   # rolling dup of "few"@1.2 → dropped
                {"utf8": " years", "tOffsetMs": 300},
            ]},
        ]}
        words = captions._parse_auto_words(data)
        self.assertEqual(["a", "few", "years"], [w.text for w in words])
        self.assertAlmostEqual(1.0, words[0].start, places=3)
        self.assertAlmostEqual(1.2, words[1].start, places=3)
        # end of a word == start of next
        self.assertAlmostEqual(words[1].start, words[0].end, places=3)
        # last word gets +0.30s tail
        self.assertAlmostEqual(words[2].start + 0.30, words[2].end, places=3)
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `uv run python -m unittest tests.test_captions.ParseAutoWordsTests -v`
Expected: FAIL (`has no attribute '_parse_auto_words'`).

- [ ] **Step 3: Implement**

```python
def _parse_auto_words(data: dict) -> list[_Word]:
    raw: list[tuple[str, float]] = []
    for ev in data.get("events") or []:
        segs = ev.get("segs")
        if not segs:
            continue
        base = ev.get("tStartMs") or 0
        for s in segs:
            w = (s.get("utf8") or "").strip()
            if not w:
                continue
            abs_start = (base + (s.get("tOffsetMs") or 0)) / 1000.0
            raw.append((w, abs_start))
    raw.sort(key=lambda x: x[1])

    deduped: list[tuple[str, float]] = []
    for text, t in raw:
        if deduped and deduped[-1][0].casefold() == text.casefold() and abs(t - deduped[-1][1]) < 0.10:
            continue
        deduped.append((text, t))

    words: list[_Word] = []
    for i, (text, t) in enumerate(deduped):
        end = deduped[i + 1][1] if i + 1 < len(deduped) else t + 0.30
        words.append(_Word(text=text, start=t, end=max(end, t)))
    return words
```

- [ ] **Step 4: Chạy test — kỳ vọng PASS**

Run: `uv run python -m unittest tests.test_captions.ParseAutoWordsTests -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/captions.py tests/test_captions.py
git commit -m "feat(captions): parse word-level auto-captions with rolling dedup

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Chunk + re-align (`_norm`, `_chunk_words`, `_align_chunk`)

**Files:** Modify `pipeline/captions.py` · `tests/test_captions.py`

- [ ] **Step 1: Viết test thất bại**

```python
class AlignTests(unittest.TestCase):
    def _w(self, items):
        return [captions._Word(t, s, e) for t, s, e in items]

    def test_chunk_breaks_on_gap(self):
        words = self._w([("a", 0.0, 0.5), ("b", 0.5, 1.0), ("c", 4.0, 4.5)])
        chunks = captions._chunk_words(words, max_words=100, max_gap=2.0)
        self.assertEqual([["a", "b"], ["c"]], [[w.text for w in c] for c in chunks])

    def test_align_splits_sentences_on_terminal_punct(self):
        words = self._w([("a", 0.0, 1.0), ("few", 1.0, 2.0), ("years", 2.0, 3.0),
                         ("hello", 3.0, 4.0), ("there", 4.0, 5.0)])
        punctuated = "A few years. Hello there?"
        segs = captions._align_chunk(words, punctuated)
        self.assertEqual(2, len(segs))
        self.assertEqual("A few years.", segs[0].text)
        self.assertAlmostEqual(0.0, segs[0].start, places=3)
        self.assertAlmostEqual(3.0, segs[0].end, places=3)
        self.assertEqual("Hello there?", segs[1].text)
        self.assertAlmostEqual(3.0, segs[1].start, places=3)
        self.assertAlmostEqual(5.0, segs[1].end, places=3)

    def test_align_returns_none_on_large_token_mismatch(self):
        words = self._w([("a", 0.0, 1.0), ("b", 1.0, 2.0)])
        # LLM returned far more tokens than words → signal fallback
        segs = captions._align_chunk(words, "a b c d e f g h.")
        self.assertIsNone(segs)
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `uv run python -m unittest tests.test_captions.AlignTests -v`
Expected: FAIL (`has no attribute '_chunk_words'`).

- [ ] **Step 3: Implement**

```python
def _norm(tok: str) -> str:
    return _PUNCT_STRIP.sub("", tok).casefold()


def _chunk_words(words: list[_Word], max_words: int = 120, max_gap: float = 2.0) -> list[list[_Word]]:
    chunks: list[list[_Word]] = []
    cur: list[_Word] = []
    for i, w in enumerate(words):
        cur.append(w)
        gap_break = i + 1 < len(words) and (words[i + 1].start - w.end) > max_gap
        if len(cur) >= max_words or gap_break:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def _align_chunk(words: list[_Word], punctuated: str) -> list[Segment] | None:
    if not words:
        return []
    tokens = punctuated.split()
    if not tokens:
        return None
    if abs(len(tokens) - len(words)) / max(1, len(words)) > 0.15:
        return None  # LLM altered the word stream → caller falls back

    segs: list[Segment] = []
    cur_words: list[_Word] = []
    cur_text: list[str] = []
    wi = 0
    for tok in tokens:
        if wi < len(words):
            cur_words.append(words[wi])
            wi += 1
        cur_text.append(tok)
        if _SENT_END.search(tok) and _is_sentence_end(tok):
            text = " ".join(cur_text).strip()
            if cur_words and text:
                segs.append(Segment(id=len(segs), start=cur_words[0].start,
                                    end=cur_words[-1].end, text=text))
            cur_words, cur_text = [], []
    if cur_words and cur_text:
        text = " ".join(cur_text).strip()
        if text:
            segs.append(Segment(id=len(segs), start=cur_words[0].start,
                                end=cur_words[-1].end, text=text))
    return segs
```

- [ ] **Step 4: Chạy test — kỳ vọng PASS**

Run: `uv run python -m unittest tests.test_captions.AlignTests -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/captions.py tests/test_captions.py
git commit -m "feat(captions): chunking + sentence re-align with mismatch guard

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Prompt + LLM restore (`prompts/restore_punctuation.yaml`, `_restore_punctuation`)

**Files:** Create `prompts/restore_punctuation.yaml` · Modify `pipeline/captions.py` · `tests/test_captions.py`

- [ ] **Step 1: Tạo prompt `prompts/restore_punctuation.yaml`**

```yaml
system: |
  You restore punctuation and capitalization in a raw speech-to-text transcript.
  STRICT RULES:
  - Add sentence-ending punctuation (. ! ?), commas, and correct capitalization.
  - DO NOT add, remove, reorder, translate, or change any word. Keep every word
    exactly as given, in the same order. Only insert punctuation and fix casing.
  Respond with JSON only: {"text": "<punctuated transcript>"}.

user: |
  Source language: {source_language}
  Raw transcript:
  {text}
```

- [ ] **Step 2: Viết test thất bại** (mock LLM client)

```python
from unittest.mock import MagicMock


def _fake_client(content):
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    resp.usage = MagicMock(prompt_tokens=5, completion_tokens=7)
    client.chat.completions.create.return_value = resp
    return client


class RestorePunctuationTests(unittest.TestCase):
    def test_returns_punctuated_text_and_tracks_usage(self):
        from pipeline.costs import CostTracker
        client = _fake_client('{"text": "A few years."}')
        tracker = CostTracker()
        out = captions._restore_punctuation("a few years", client, "English", tracker)
        self.assertEqual("A few years.", out)
        client.chat.completions.create.assert_called_once()
```

- [ ] **Step 3: Chạy test — kỳ vọng FAIL**

Run: `uv run python -m unittest tests.test_captions.RestorePunctuationTests -v`
Expected: FAIL (`has no attribute '_restore_punctuation'`).

- [ ] **Step 4: Implement**

```python
def _together_extra() -> dict:
    if get_translation_provider(_conf.get()) == "together":
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {}


def _restore_punctuation(text: str, client, source_language: str,
                         tracker: CostTracker | None = None) -> str:
    cfg = _conf.get()
    model = get_translation_model(cfg)
    max_retries = cfg["translation"].get("max_retries", 3)
    system_msg = _prompts.load("restore_punctuation", "system", source_language=source_language)
    user_msg = _prompts.load("restore_punctuation", "user",
                             source_language=source_language, text=text)
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "restore_punctuation",
                                    "strict": True, "schema": _PUNCT_SCHEMA},
                },
                **_together_extra(),
            )
            if tracker and getattr(response, "usage", None):
                tracker.add_llm_usage(response.usage.prompt_tokens or 0,
                                      response.usage.completion_tokens or 0)
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)["text"]
        except Exception as exc:  # transient API / parse error
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                raise
```

- [ ] **Step 5: Chạy test — kỳ vọng PASS**

Run: `uv run python -m unittest tests.test_captions.RestorePunctuationTests -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add prompts/restore_punctuation.yaml pipeline/captions.py tests/test_captions.py
git commit -m "feat(captions): LLM punctuation restoration prompt + client call

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Restore + align orchestration (`_restore_punctuation_and_align`)

**Files:** Modify `pipeline/captions.py` · `tests/test_captions.py`

- [ ] **Step 1: Viết test thất bại**

```python
class RestoreAndAlignTests(unittest.TestCase):
    def _w(self, items):
        return [captions._Word(t, s, e) for t, s, e in items]

    def test_happy_path_aligns_to_word_timestamps(self):
        words = self._w([("a", 0.0, 1.0), ("few", 1.0, 2.0), ("years", 2.0, 3.0)])
        client = _fake_client('{"text": "A few years."}')
        segs = captions._restore_punctuation_and_align(words, client, "English", None)
        self.assertEqual(["A few years."], [s.text for s in segs])
        self.assertAlmostEqual(0.0, segs[0].start, places=3)
        self.assertAlmostEqual(3.0, segs[0].end, places=3)

    def test_fallback_one_segment_when_llm_alters_tokens(self):
        words = self._w([("a", 0.0, 1.0), ("b", 1.0, 2.0)])
        # token count far off → _align_chunk returns None → fallback to one seg
        client = _fake_client('{"text": "a b c d e f g h i."}')
        segs = captions._restore_punctuation_and_align(words, client, "English", None)
        self.assertEqual(1, len(segs))
        self.assertEqual("a b c d e f g h i.", segs[0].text)
        self.assertAlmostEqual(0.0, segs[0].start, places=3)
        self.assertAlmostEqual(2.0, segs[0].end, places=3)

    def test_fallback_uses_raw_when_llm_raises(self):
        words = self._w([("a", 0.0, 1.0), ("b", 1.0, 2.0)])
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("boom")
        # max_retries from config; ensure it eventually falls back to raw text
        segs = captions._restore_punctuation_and_align(words, client, "English", None)
        self.assertEqual(1, len(segs))
        self.assertEqual("a b", segs[0].text)
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `uv run python -m unittest tests.test_captions.RestoreAndAlignTests -v`
Expected: FAIL (`has no attribute '_restore_punctuation_and_align'`).

- [ ] **Step 3: Implement**

```python
def _restore_punctuation_and_align(words: list[_Word], client, source_language: str,
                                   tracker: CostTracker | None = None) -> list[Segment]:
    all_segs: list[Segment] = []
    for chunk in _chunk_words(words):
        raw_text = " ".join(w.text for w in chunk)
        try:
            punct = _restore_punctuation(raw_text, client, source_language, tracker)
        except Exception as exc:
            print(f"      [captions] punctuation restore failed: {exc}")
            punct = None
        segs = _align_chunk(chunk, punct) if punct else None
        if not segs:  # None (mismatch) or empty → fallback 🅑: one segment per chunk
            text = (punct or raw_text).strip()
            segs = [Segment(id=0, start=chunk[0].start, end=chunk[-1].end, text=text)]
        all_segs.extend(segs)
    for i, s in enumerate(all_segs):
        s.id = i
    return all_segs
```

- [ ] **Step 4: Chạy test — kỳ vọng PASS**

Run: `uv run python -m unittest tests.test_captions.RestoreAndAlignTests -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/captions.py tests/test_captions.py
git commit -m "feat(captions): restore+align per chunk with robust fallback

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Public entry (`fetch_source_captions`)

**Files:** Modify `pipeline/captions.py` · `tests/test_captions.py`

- [ ] **Step 1: Viết test thất bại** (mock yt_dlp + json3 download)

```python
from unittest.mock import patch


class FetchSourceCaptionsTests(unittest.TestCase):
    def _info(self):
        return {
            "language": "en",
            "subtitles": {"en": [{"ext": "json3", "url": "MAN_URL"}]},
            "automatic_captions": {},
        }

    def test_manual_path_returns_segments(self):
        manual_json = {"events": [
            {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "Hello there."}]},
        ]}
        fake_ydl = MagicMock()
        fake_ydl.__enter__.return_value.extract_info.return_value = self._info()
        with patch("pipeline.captions.yt_dlp_module", create=True), \
             patch("pipeline.captions._open_ydl", return_value=fake_ydl), \
             patch("pipeline.captions._download_json3", return_value=manual_json):
            segs = captions.fetch_source_captions("https://x/y", "English")
        self.assertEqual(["Hello there."], [s.text for s in segs])

    def test_returns_none_when_no_track(self):
        info = {"language": "en", "subtitles": {}, "automatic_captions": {}}
        fake_ydl = MagicMock()
        fake_ydl.__enter__.return_value.extract_info.return_value = info
        with patch("pipeline.captions._open_ydl", return_value=fake_ydl):
            self.assertIsNone(captions.fetch_source_captions("https://x/y", "English"))

    def test_returns_none_on_extract_error(self):
        with patch("pipeline.captions._open_ydl", side_effect=RuntimeError("net")):
            self.assertIsNone(captions.fetch_source_captions("https://x/y", "English"))
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `uv run python -m unittest tests.test_captions.FetchSourceCaptionsTests -v`
Expected: FAIL (`has no attribute '_open_ydl'` / `fetch_source_captions`).

- [ ] **Step 3: Implement** (`_open_ydl` được tách riêng để test mock dễ)

```python
def _open_ydl():
    import yt_dlp
    opts = {"skip_download": True, "quiet": True, "no_warnings": True,
            "writesubtitles": True, "writeautomaticsub": True, "noplaylist": True}
    return yt_dlp.YoutubeDL(opts)


def _download_json3(url: str, *, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_source_captions(url: str, source_language: str = "auto-detect", *,
                          prefer_manual: bool = True, llm_client=None,
                          tracker: CostTracker | None = None) -> list[Segment] | None:
    """Return Segment[] from the URL's source-language captions, or None.

    None means no usable caption was found — the caller should run Whisper.
    """
    try:
        with _open_ydl() as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        print(f"      [captions] extract_info failed: {exc}")
        return None

    src = _source_lang_code(info, source_language)
    track = _select_track(info, src)
    if track is None:
        print("      [captions] no suitable source-language caption track")
        return None

    try:
        data = _download_json3(track.url)
    except Exception as exc:
        print(f"      [captions] json3 download failed: {exc}")
        return None

    if track.kind == "manual":
        segs = _parse_manual(data)
    else:
        words = _parse_auto_words(data)
        if not words:
            return None
        if llm_client is None:
            llm_client = make_translation_client(_conf.get())
        segs = _restore_punctuation_and_align(words, llm_client, source_language, tracker)

    if not segs:
        return None
    print(f"      [captions] using {track.kind} captions [{track.lang}] — {len(segs)} segments")
    return segs
```

> Note: test patches `_open_ydl` to return a context-manager mock, so `import yt_dlp` is never executed in tests. The `yt_dlp_module` patch in the first test is a harmless no-op guard (create=True).

- [ ] **Step 4: Chạy test — kỳ vọng PASS**

Run: `uv run python -m unittest tests.test_captions -v`
Expected: PASS (toàn bộ test_captions).

- [ ] **Step 5: Commit**

```bash
git add pipeline/captions.py tests/test_captions.py
git commit -m "feat(captions): fetch_source_captions entry point with Whisper fallback

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Orchestrator — `segments_override` + `prefer_source_captions`

**Files:** Modify `pipeline/orchestrator.py` · `tests/test_orchestrator_artifacts.py`

- [ ] **Step 1: Viết test thất bại** (thêm vào `tests/test_orchestrator_artifacts.py`)

```python
    def test_segments_override_skips_transcription(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.mp4"
            out.write_bytes(b"video")
            tts = Path(tmp) / "seg.wav"
            tts.write_bytes(b"wav")

            override = [Segment(id=0, start=0.0, end=1.0, text="From caption")]
            translated = [Segment(id=0, start=0.0, end=1.0, text="Tu phu de")]

            with patch("pipeline.orchestrator.make_translation_client"), \
                 patch("pipeline.orchestrator.make_transcription_client") as mk_tc, \
                 patch("pipeline.orchestrator.extract_audio") as extract, \
                 patch("pipeline.orchestrator.get_video_duration", return_value=1.0), \
                 patch("pipeline.orchestrator.ensure_video_input", return_value="input.mp4"), \
                 patch("pipeline.orchestrator.transcribe") as transcribe_mock, \
                 patch("pipeline.orchestrator.translate_segments", return_value=translated), \
                 patch("pipeline.orchestrator.synthesize_segments", return_value=[str(tts)]), \
                 patch("pipeline.orchestrator.prepare_merge", return_value=object()), \
                 patch("pipeline.orchestrator.build_gap_chunks"), \
                 patch("pipeline.orchestrator.build_aligned_video", return_value=translated):
                result = dub_video(
                    "input.mp4", str(out),
                    DubOptions(target_language="Vietnamese", subtitles=False),
                    segments_override=override,
                )

            transcribe_mock.assert_not_called()
            extract.assert_not_called()
            mk_tc.assert_not_called()
            self.assertEqual("Tu phu de", result.aligned_segments[0].text)
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `uv run python -m unittest tests.test_orchestrator_artifacts.OrchestratorArtifactTests.test_segments_override_skips_transcription -v`
Expected: FAIL (`dub_video() got an unexpected keyword argument 'segments_override'`).

- [ ] **Step 3: Implement — `DubOptions`** (thêm field, sau dòng `burn_subtitles: bool = False` ở `pipeline/orchestrator.py`)

```python
    prefer_source_captions: bool = True   # URL inputs: prefer YouTube captions over Whisper
```

- [ ] **Step 4: Implement — chữ ký `dub_video`** (thêm keyword param vào danh sách, ví dụ sau `tracker: CostTracker | None = None,`)

```python
    segments_override: list[Segment] | None = None,
```

- [ ] **Step 5: Implement — defer transcription client + nhánh override**

Thay khối tạo client + bước 1–2 hiện tại. **Trước:**

```python
    translation_client = make_translation_client(
        cfg,
        together_key_override=opts.together_api_key,
        openai_key_override=opts.openai_api_key,
    )
    transcription_client = make_transcription_client(
        cfg,
        together_key_override=opts.together_api_key,
        openai_key_override=opts.openai_api_key,
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="vidtrans_"))
    try:
        tracker.start_timer()

        _check_cancel(is_cancelled)
        _emit(on_progress, 1, "Extracting audio…")
        audio_path = extract_audio(input_path, str(tmp_dir / "audio.wav"))
        total_duration = get_video_duration(input_path)
        video_input_path = ensure_video_input(input_path, str(tmp_dir / "audio_input.mp4"))
        tracker.audio_minutes = total_duration / 60.0
        tracker.record_step("Audio extraction")

        _check_cancel(is_cancelled)
        _emit(on_progress, 2, f"Transcribing with Whisper Large v3… (duration: {total_duration:.0f}s)")
        segments = transcribe(audio_path, transcription_client)
        tracker.record_step("Transcription (Whisper)")
```

**Sau:**

```python
    translation_client = make_translation_client(
        cfg,
        together_key_override=opts.together_api_key,
        openai_key_override=opts.openai_api_key,
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="vidtrans_"))
    try:
        tracker.start_timer()

        _check_cancel(is_cancelled)
        total_duration = get_video_duration(input_path)
        video_input_path = ensure_video_input(input_path, str(tmp_dir / "audio_input.mp4"))
        tracker.audio_minutes = total_duration / 60.0

        if segments_override is not None:
            _emit(on_progress, 2, f"Using source captions ({len(segments_override)} segments)…")
            segments = segments_override
            tracker.record_step("Source captions")
        else:
            _emit(on_progress, 1, "Extracting audio…")
            audio_path = extract_audio(input_path, str(tmp_dir / "audio.wav"))
            tracker.record_step("Audio extraction")
            transcription_client = make_transcription_client(
                cfg,
                together_key_override=opts.together_api_key,
                openai_key_override=opts.openai_api_key,
            )
            _check_cancel(is_cancelled)
            _emit(on_progress, 2, f"Transcribing with Whisper Large v3… (duration: {total_duration:.0f}s)")
            segments = transcribe(audio_path, transcription_client)
            tracker.record_step("Transcription (Whisper)")
```

- [ ] **Step 6: Chạy test — kỳ vọng PASS** (cả test cũ lẫn mới)

Run: `uv run python -m unittest tests.test_orchestrator_artifacts -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add pipeline/orchestrator.py tests/test_orchestrator_artifacts.py
git commit -m "feat(orchestrator): segments_override bypasses Whisper; defer STT client

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: CLI — `--no-source-captions` + gọi caption (`main.py`)

**Files:** Modify `main.py` · `tests/test_cli_url_input.py`

- [ ] **Step 1: Viết test thất bại** (thêm vào `tests/test_cli_url_input.py`)

```python
    def test_url_fetches_source_captions_and_passes_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            downloaded = Path(tmp) / "input.mp4"
            downloaded.write_bytes(b"video")
            from pipeline.transcriber import Segment
            caps = [Segment(id=0, start=0.0, end=1.0, text="From caption")]

            with patch("main.download_url_to_file", return_value=downloaded), \
                 patch("main.fetch_source_captions", return_value=caps) as fetch, \
                 patch("main.dub_video") as dub, \
                 patch("main.resolve_style", return_value=type("Style", (), {
                     "name": "standard", "description": ""})()):
                dub.return_value = type("Result", (), {
                    "original_audio_path": None, "subtitle_paths": {},
                    "burned_video_path": None, "transcript_path": None,
                    "output_video_path": "out.mp4",
                    "cost_tracker": type("T", (), {"print_summary": lambda self: None})(),
                    "steps": [],
                })()
                main.translate_video(
                    "https://www.youtube.com/watch?v=test", "out.mp4", "Vietnamese",
                )
            fetch.assert_called_once()
            self.assertEqual(caps, dub.call_args.kwargs["segments_override"])
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `uv run python -m unittest tests.test_cli_url_input.CliUrlInputTests.test_url_fetches_source_captions_and_passes_override -v`
Expected: FAIL (`module 'main' has no attribute 'fetch_source_captions'`).

- [ ] **Step 3: Implement — import + tham số `translate_video`**

Thêm import gần các import pipeline trong `main.py`:

```python
from pipeline.captions import fetch_source_captions
```

Thêm param vào chữ ký `translate_video(...)` (sau `timings_out`):

```python
    prefer_source_captions: bool = True,
```

- [ ] **Step 4: Implement — lấy caption trong khối URL**

Trong `translate_video`, **trước** thay khối:

```python
        effective_input = input_path
        if is_url(input_path):
            tmp_download_dir = tempfile.mkdtemp(prefix="violin_url_")
            print(f"\n[0/5] Downloading media URL…")
            effective_input = str(download_url_to_file(input_path, tmp_download_dir))

        result = dub_video(
            effective_input,
            output_path,
            opts,
            output_srt_path=srt_path,
            burned_video_path=str(out_p.with_stem(out_p.stem + "_subtitled")) if burn_subtitles else None,
            original_audio_path=orig_audio_path,
            on_progress=lambda step, msg: print(f"\n[{step}/5] {msg}"),
        )
```

**Sau:**

```python
        effective_input = input_path
        segments_override = None
        if is_url(input_path):
            tmp_download_dir = tempfile.mkdtemp(prefix="violin_url_")
            print(f"\n[0/5] Downloading media URL…")
            effective_input = str(download_url_to_file(input_path, tmp_download_dir))
            if prefer_source_captions:
                print(f"\n[0/5] Checking for source captions…")
                segments_override = fetch_source_captions(input_path, source_language)

        result = dub_video(
            effective_input,
            output_path,
            opts,
            output_srt_path=srt_path,
            burned_video_path=str(out_p.with_stem(out_p.stem + "_subtitled")) if burn_subtitles else None,
            original_audio_path=orig_audio_path,
            on_progress=lambda step, msg: print(f"\n[{step}/5] {msg}"),
            segments_override=segments_override,
        )
```

- [ ] **Step 5: Implement — CLI flag + truyền xuống**

Thêm argument trong `main()` (gần các flag khác):

```python
    parser.add_argument(
        "--no-source-captions", action="store_true",
        help="Always transcribe with Whisper instead of reusing the video's captions"
    )
```

Tính giá trị hiệu lực + truyền vào `translate_video(...)` ở cuối `main()`. Thêm trước lời gọi:

```python
    prefer_source_captions = (
        pipeline_config.get()["transcription"].get("prefer_source_captions", True)
        and not args.no_source_captions
    )
```

Và thêm tham số vào lời gọi `translate_video(...)`:

```python
        prefer_source_captions=prefer_source_captions,
```

- [ ] **Step 6: Chạy test — kỳ vọng PASS** (cả test cũ lẫn mới)

Run: `uv run python -m unittest tests.test_cli_url_input -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add main.py tests/test_cli_url_input.py
git commit -m "feat(cli): prefer source captions for URL input; --no-source-captions

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: API worker — lấy caption trong `_run_url_job`

**Files:** Modify `api/worker.py` · Test `tests/test_worker_url_captions.py` (mới)

- [ ] **Step 1: Viết test thất bại** (`tests/test_worker_url_captions.py`)

```python
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from api import worker


class WorkerUrlCaptionsTests(unittest.TestCase):
    def test_url_job_passes_caption_segments_to_run_job(self):
        from pipeline.transcriber import Segment
        caps = [Segment(id=0, start=0.0, end=1.0, text="cap")]
        params = {"language": "Vietnamese", "voice": "", "source_language": "auto-detect",
                  "subtitles": True, "style": "standard", "voiceover": True,
                  "prefer_source_captions": True}
        with patch("api.worker.update_status"), \
             patch("api.worker.append_progress"), \
             patch("api.worker._download_url", return_value=Path("/tmp/input.mp4")), \
             patch("api.worker.fetch_source_captions", return_value=caps) as fetch, \
             patch("api.worker._run_job") as run_job:
            worker._run_url_job("job1", params, "https://youtu.be/x")
        fetch.assert_called_once()
        self.assertEqual(caps, run_job.call_args.kwargs["segments_override"])
```

- [ ] **Step 2: Chạy test — kỳ vọng FAIL**

Run: `uv run python -m unittest tests.test_worker_url_captions -v`
Expected: FAIL (`module 'api.worker' has no attribute 'fetch_source_captions'`).

- [ ] **Step 3: Implement — import + `_run_job` nhận override**

Thêm import trong `api/worker.py`:

```python
from pipeline.captions import fetch_source_captions
```

Thêm tham số `segments_override` cho `_run_job` (sau `elevenlabs_key_override`):

```python
    segments_override=None,
```

Và truyền vào `dub_video(...)` trong `_run_job` (thêm kwarg):

```python
            segments_override=segments_override,
```

- [ ] **Step 4: Implement — `_run_url_job` lấy caption sau khi tải**

**Trước:**

```python
    try:
        _download_url(job_id, url)
    except Exception as exc:
        update_status(job_id, JobStatus.failed, f"Download failed: {exc}")
        return

    _run_job(job_id, params, together_key_override, openai_key_override, elevenlabs_key_override)
```

**Sau:**

```python
    try:
        _download_url(job_id, url)
    except Exception as exc:
        update_status(job_id, JobStatus.failed, f"Download failed: {exc}")
        return

    segments_override = None
    if params.get("prefer_source_captions", True):
        append_progress(job_id, 1, TOTAL_STEPS, "Checking for source captions…")
        segments_override = fetch_source_captions(url, params.get("source_language", "auto-detect"))

    _run_job(job_id, params, together_key_override, openai_key_override,
             elevenlabs_key_override, segments_override=segments_override)
```

- [ ] **Step 5: Chạy test — kỳ vọng PASS**

Run: `uv run python -m unittest tests.test_worker_url_captions -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add api/worker.py tests/test_worker_url_captions.py
git commit -m "feat(api): prefer source captions in URL jobs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: API request field + config default

**Files:** Modify `api/models.py`, `api/routes/jobs.py`, `config/default.yaml` · Test `tests/test_jobs_history_route.py` (hoặc test route from-url hiện có)

- [ ] **Step 1: Tìm model request from-url + thêm field**

Run: `uv run python -c "import api.models, inspect; print(inspect.getsourcefile(api.models))"` rồi mở file. Trong model dùng cho `POST /jobs/from-url` (thường `UrlJobRequest` hoặc tương tự), thêm field:

```python
    prefer_source_captions: bool = True
```

> Nếu request from-url đọc form-data trong `api/routes/jobs.py` thay vì Pydantic model, thêm tham số `prefer_source_captions: bool = Form(True)` vào hàm route và đưa vào `params` dict gửi cho `submit_url_job`.

- [ ] **Step 2: Viết test thất bại** (`tests/test_url_job_prefer_captions.py` mới)

```python
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from api.app import app


class UrlJobPreferCaptionsTests(unittest.TestCase):
    def test_from_url_defaults_prefer_source_captions_true(self):
        with patch("api.routes.jobs.submit_url_job") as submit, \
             patch("api.routes.jobs.create_job"):
            client = TestClient(app)
            client.post("/jobs/from-url", data={"url": "https://youtu.be/x",
                                                "language": "Vietnamese"})
        # params is passed positionally or by kw depending on route; assert present
        called = submit.call_args
        params = called.kwargs.get("params") or called.args[1]
        self.assertTrue(params.get("prefer_source_captions", True))
```

> Điều chỉnh tên route/tham số cho khớp `api/routes/jobs.py` thực tế khi mở file ở Step 1.

- [ ] **Step 3: Chạy test — kỳ vọng FAIL → Implement → PASS**

Run: `uv run python -m unittest tests.test_url_job_prefer_captions -v`
Sửa `api/routes/jobs.py` để đưa `prefer_source_captions` vào `params`. Chạy lại → Expected: PASS.

- [ ] **Step 4: Thêm config default** — sửa `config/default.yaml`, mục `transcription`:

```yaml
transcription:
  chunk_seconds: 600
  parallel_workers: 2
  prefer_source_captions: true   # URL inputs reuse YouTube captions when available
```

- [ ] **Step 5: Chạy toàn bộ test**

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS toàn bộ.

- [ ] **Step 6: Commit**

```bash
git add api/models.py api/routes/jobs.py config/default.yaml tests/test_url_job_prefer_captions.py
git commit -m "feat(api): prefer_source_captions request field + config default

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Web UI checkbox (verify thủ công)

**Files:** Modify `api/static/index.html`

- [ ] **Step 1: Thêm checkbox** trong khu vực tuỳ chọn job (gần checkbox subtitle), Alpine state mặc định `true`:

```html
<label class="opt">
  <input type="checkbox" x-model="preferSourceCaptions">
  Ưu tiên phụ đề YouTube (nhanh hơn)
</label>
```

Trong Alpine data: thêm `preferSourceCaptions: true,` và khi build form-data cho `POST /jobs/from-url`, thêm:

```javascript
form.append('prefer_source_captions', this.preferSourceCaptions);
```

- [ ] **Step 2: Verify thủ công**

Run: `uv run run_api.py` → mở `http://127.0.0.1:8000`, dán URL YouTube, để checkbox bật → submit → log job hiện "Using source captions (...)". Bỏ checkbox → log hiện "Transcribing with Whisper".

- [ ] **Step 3: Commit**

```bash
git add api/static/index.html
git commit -m "feat(ui): prefer-source-captions checkbox for URL jobs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Verify cuối (toàn bộ)

- [ ] `uv run python -m unittest discover -s tests -v` → tất cả PASS.
- [ ] Smoke thật (tuỳ chọn, cần `OLLAMA_API_KEY` + `uv sync --extra local`):
  `uv run main.py "https://www.youtube.com/watch?v=<id-có-caption>" out_vi.mp4 --language Vietnamese --config config/local_mac.yaml --subtitle-formats srt,txt`
  → log hiện "Using source captions (N segments)…", không gọi Whisper; kiểm tra `out_vi.srt` khớp giọng đọc.
  Lặp lại với `--no-source-captions` → log hiện "Transcribing with Whisper".

---

## Self-Review (đã chạy khi viết plan)

- **Spec coverage:** §5 module → Task 1–7 · §6 chọn track → Task 1 · §7 manual → Task 2 · §8a–e auto/restore/align/fallback → Task 3,4,5,6 · §9 tích hợp → Task 8,9,10,11,12 · §10 fallback phân tầng → Task 7 (None→Whisper) + Task 6 (🅑) · §11 testing → mỗi task có test · §13 tham số → nhúng trong code (0.10s dedup, +0.30s tail, 2.0s/120 từ chunk, 15% mismatch, json3).
- **Placeholder scan:** không có TBD/TODO; code đầy đủ mỗi step. Task 11 có 1 chỗ phụ thuộc hình dạng route thực tế (Pydantic vs Form) — đã ghi rõ cả hai nhánh + lệnh để xác định.
- **Type consistency:** `_Word(text,start,end)`, `_TrackRef(kind,lang,url)`, `Segment(id,start,end,text,speaker)`, `fetch_source_captions(...)→list[Segment]|None`, `dub_video(...,segments_override=...)` — tên/chữ ký nhất quán xuyên suốt Task 1→12.
