# Local EN→VI Dubbing — M2 (pipelining, sentence splitting, fill slow-down, multi-speaker) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trên nền M1 (nhánh `feat/local-vi-dubbing`, 131 test): (1) dịch batch N+1 song song với TTS batch N để dùng cả 2 GPU; (2) tách câu ASR quá dài theo dấu phẩy/khoảng lặng; (3) kéo chậm nhẹ unit lấp dưới 60 % ngân sách; (4) nhiều người nói: diarization theo câu + gán giọng preset theo speaker/giới tính.

**Architecture:** Không đổi contract của các module M1. Thêm: `transcriber.split_long_segments` (áp sau merge); `fitter` có `tempo` (ffmpeg atempo < 1 trên clip) và `run_pipelined` (pool dịch 2 luồng + consumer TTS); `pipeline/diarizer.py` (backend `ecapa`: embedding ECAPA của speechbrain trên từng câu + clustering; backend `pyannote` tuỳ chọn); `voices` gán giọng theo speaker (round-robin danh sách preset, hoặc theo giới tính đo bằng F0 numpy); orchestrator nối `--speakers auto`.

**Tech Stack:** như M1 + `speechbrain>=1.0`, `scipy` (extra `local-gpu`); pyannote tuỳ chọn (cần HF token).

**Spec:** `docs/superpowers/specs/2026-09-02-local-vi-dubbing-design.md` (mục Diarization/Voices + Amendment M2 sẽ thêm).

## Global Constraints

- Test: `uv run python -m pytest -q`; 131 test hiện có phải pass nguyên trạng; mọi khoá config mới mặc định tắt trong `config/default.yaml`, bật trong 2 preset local.
- Không đổi chữ ký public của M1 (`fit_audio`, `apply_units`, `translate_segments`, `synthesize_batch`, `build_time_map`, `split_into_cues`).
- Không bao giờ tăng tốc giọng trong fitter; kéo chậm tối đa `fit.min_tempo` (0.85, atempo giữ cao độ).
- Mac không có torch → diarization chỉ trên preset GPU (`--speakers 1` mặc định trên Mac); code phải import speechbrain/torch lazily.
- Commit trailer như M1: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` + `Claude-Session: https://claude.ai/code/session_01HjBXyTtKKvNCAwEkCRHgNs`.

---

### Task 20: Tách câu ASR quá dài theo dấu phẩy / khoảng lặng (áp sau merge)

**Files:** Modify `pipeline/transcriber.py` (merge giữ `words`; thêm `split_long_segments`), `pipeline/orchestrator.py` (gọi sau `merge_continuous_segments`), `config/default.yaml` (`transcription.max_sentence_seconds: 0`, `min_piece_seconds: 2.5`), `config/local_mac.yaml` + `config/local_gpu.yaml` (`max_sentence_seconds: 12`). Test: `tests/test_transcriber_split_long.py`.

**Interfaces:** `split_long_segments(segments: list[Segment], max_seconds: float, min_piece_seconds: float = 2.5) -> list[Segment]` — chỉ tách segment có `words`; chọn điểm cắt tại ranh giới từ i (1..n-1) sao cho hai nửa ≥ `min_piece_seconds`; ưu tiên từ kết thúc bằng `, ; : . ! ?`, trong số đó chọn gần giữa nhất; nếu không có thì chọn khoảng lặng lớn nhất `words[i].start - words[i-1].end`; đệ quy tới khi mọi mảnh ≤ `max_seconds`; text/start/end/words dựng lại từ words; id đánh lại; `speaker`/`source_text` giữ. `merge_continuous_segments`/`_absorb_short_segments` nối `words` khi cả hai có (else `None`).

- [ ] **Step 1: Test thất bại** — `tests/test_transcriber_split_long.py`:

```python
from pipeline import config as pipeline_config
from pipeline.transcriber import Segment, merge_continuous_segments, split_long_segments


def _sent(words, speaker="SPEAKER_00"):
    ws = [[w, s, e] for w, s, e in words]
    return Segment(id=0, start=ws[0][1], end=ws[-1][2], text=" ".join(w for w, _, _ in words), speaker=speaker, words=ws)


def test_splits_at_comma_nearest_middle():
    words = [(f"w{i}" + ("," if i == 9 else ""), i * 1.0, i * 1.0 + 0.8) for i in range(20)]   # 0–19.8 s, comma after w9
    out = split_long_segments([_sent(words)], max_seconds=12.0)
    assert [s.text.split()[0] for s in out] == ["w0", "w10"]
    assert out[0].end == 9.8 and out[1].start == 10.0
    assert out[0].words[-1][0] == "w9," and [s.id for s in out] == [0, 1]


def test_splits_at_largest_gap_without_punctuation():
    words = [(f"w{i}", i * 1.0 + (2.0 if i >= 12 else 0.0), i * 1.0 + 0.8 + (2.0 if i >= 12 else 0.0)) for i in range(20)]
    out = split_long_segments([_sent(words)], max_seconds=12.0)
    assert len(out) == 2 and out[0].words[-1][0] == "w11" and out[1].words[0][0] == "w12"


def test_short_or_wordless_segments_untouched():
    short = _sent([("a", 0.0, 0.5), ("b.", 0.6, 1.0)])
    nowords = Segment(id=0, start=0.0, end=30.0, text="x " * 50)
    out = split_long_segments([short, nowords], max_seconds=12.0)
    assert out[0].text == "a b." and out[1].end == 30.0 and len(out) == 2


def test_min_piece_respected_and_recursive():
    words = [(f"w{i}" + ("," if i in (1, 15) else ""), i * 1.0, i * 1.0 + 0.8) for i in range(30)]   # 30 s; comma after w1 (too early) and w15
    out = split_long_segments([_sent(words)], max_seconds=12.0, min_piece_seconds=2.5)
    assert all((s.end - s.start) <= 12.0 + 1e-6 for s in out) and all((s.end - s.start) >= 2.5 for s in out)
    assert sum(len(s.words) for s in out) == 30


def test_merge_keeps_words_when_both_have_them():
    pipeline_config.load()
    a = _sent([("Hello", 0.0, 0.4), ("there", 0.5, 0.9)])
    b = _sent([("friend.", 1.0, 1.4)])
    out = merge_continuous_segments([a, b], min_duration=0.0)
    assert len(out) == 1 and out[0].words == [["Hello", 0.0, 0.4], ["there", 0.5, 0.9], ["friend.", 1.0, 1.4]]
```

- [ ] **Step 2: RED** — `uv run python -m pytest tests/test_transcriber_split_long.py -v` → ImportError `split_long_segments`; merge test: words None.

- [ ] **Step 3: Code** — `pipeline/transcriber.py`:

```python
_CLAUSE_END = (",", ";", ":", ".", "!", "?", "。", "，")


def _concat_words(a: "Segment", b: "Segment"):
    return (a.words + b.words) if (a.words and b.words) else None


def _rebuild(words: list[list], template: "Segment") -> "Segment":
    return Segment(id=0, start=float(words[0][1]), end=float(words[-1][2]),
                   text=" ".join(w[0] for w in words).strip(), speaker=template.speaker,
                   source_text=template.source_text, words=[list(w) for w in words])


def _split_one(seg: "Segment", max_s: float, min_piece: float) -> list["Segment"]:
    words = seg.words or []
    if (seg.end - seg.start) <= max_s or len(words) < 2:
        return [seg]
    n = len(words)
    mid = (float(words[0][1]) + float(words[-1][2])) / 2.0
    cands = []
    for i in range(1, n):
        left_dur = float(words[i - 1][2]) - float(words[0][1])
        right_dur = float(words[-1][2]) - float(words[i][1])
        if left_dur < min_piece or right_dur < min_piece:
            continue
        punct = str(words[i - 1][0]).endswith(_CLAUSE_END)
        gap = float(words[i][1]) - float(words[i - 1][2])
        dist = abs(float(words[i][1]) - mid)
        cands.append((0 if punct else 1, -gap if not punct else 0.0, dist, i))
    if not cands:
        return [seg]
    cands.sort()
    i = cands[0][3]
    left, right = _rebuild(words[:i], seg), _rebuild(words[i:], seg)
    return _split_one(left, max_s, min_piece) + _split_one(right, max_s, min_piece)


def split_long_segments(segments: list["Segment"], max_seconds: float, min_piece_seconds: float = 2.5) -> list["Segment"]:
    """Split sentences longer than *max_seconds* at clause punctuation (nearest the middle) or the largest word gap."""
    if max_seconds <= 0:
        return segments
    out: list[Segment] = []
    for seg in segments:
        out.extend(_split_one(seg, max_seconds, min_piece_seconds))
    for i, s in enumerate(out):
        s.id = i
    return out
```

Sort key: ưu tiên có dấu câu (0 trước 1); trong nhóm có dấu câu chọn gần giữa nhất (gap không xét); trong nhóm không dấu câu chọn gap lớn nhất (−gap nhỏ nhất), rồi gần giữa. Trong `merge_continuous_segments` (vòng chính) và `_absorb_short_segments._join` thêm `words=_concat_words(current, seg)` / `words=_concat_words(a, b)`.

`pipeline/orchestrator.py`: ngay sau `segments = merge_continuous_segments(segments)`:

```python
        tcfg = cfg.get("transcription", {})
        segments = split_long_segments(
            segments, float(tcfg.get("max_sentence_seconds", 0) or 0), float(tcfg.get("min_piece_seconds", 2.5)),
        )
```

(import `split_long_segments` từ `.transcriber`). Config: `default.yaml` `transcription:` thêm `max_sentence_seconds: 0   # >0: split ASR sentences longer than this at clause punctuation / silences (local presets: 12)` và `min_piece_seconds: 2.5`; 2 preset: `max_sentence_seconds: 12`.

- [ ] **Step 4: GREEN + commit** `feat(transcriber): split over-long ASR sentences at clauses or silences`.

---

### Task 21: Kéo chậm nhẹ unit lấp dưới ngưỡng (`fit.min_fill`)

**Files:** Modify `pipeline/fitter.py` (`DubUnit.tempo`, `_slow_down`, áp trong `_measure`), `config/default.yaml` (`fit.min_fill: 0.0`, `fit.min_tempo: 0.85`), presets (`min_fill: 0.6`). Test: `tests/test_fitter_slowdown.py`.

**Interfaces:** `DubUnit.tempo: float = 1.0` (≤ 1; 1 = không đổi). Sau khi đo, nếu `budget_s > 0` và `tts_dur < min_fill * budget_s`: `tempo = max(min_tempo, tts_dur / (min_fill * budget_s))`; nếu `tempo < 0.995` → ghi lại clip bằng ffmpeg `atempo=tempo` (44,1 kHz mono PCM16) rồi đo lại; `strategy` thêm hậu tố `+slowed` (natural → `slowed`). `_measure(units, fcfg=None)` nhận config; `fit_audio` truyền `fcfg`.

- [ ] **Step 1: Test** — `tests/test_fitter_slowdown.py`:

```python
import numpy as np, pytest, soundfile as sf
from pipeline import fitter
from pipeline.transcriber import Segment

FCFG = dict(sec_per_syllable=0.2, overrun_tolerance=1.3, max_shorten_rounds=2, max_pause_borrow_s=0.6, margin_s=0.05, min_fill=0.6, min_tempo=0.85)


def _synth_fixed(seconds):
    def synth(text, voice, out_path, speed=1.0):
        sf.write(out_path, np.zeros(int(44100 * seconds), dtype=np.float32), 44100); return out_path
    return synth


def test_underfilled_unit_is_slowed_to_min_tempo(tmp_path):
    seg = Segment(id=0, start=0.0, end=10.0, text="ngắn")
    units = fitter.build_units([seg], [10.0], {}, "nam-1")
    fitter.fit_audio(units, _synth_fixed(3.0), str(tmp_path), FCFG)
    u = units[0]
    assert u.tempo == pytest.approx(0.85) and u.strategy == "slowed"
    assert u.tts_dur == pytest.approx(3.0 / 0.85, abs=0.05)


def test_partially_underfilled_unit_gets_intermediate_tempo(tmp_path):
    seg = Segment(id=0, start=0.0, end=10.0, text="x")
    units = fitter.build_units([seg], [10.0], {}, "nam-1")
    fitter.fit_audio(units, _synth_fixed(5.4), str(tmp_path), FCFG)    # 54 % filled → tempo 0.9
    assert units[0].tempo == pytest.approx(0.9, abs=0.01) and units[0].tts_dur == pytest.approx(6.0, abs=0.05)


def test_well_filled_unit_untouched(tmp_path):
    seg = Segment(id=0, start=0.0, end=10.0, text="x")
    units = fitter.build_units([seg], [10.0], {}, "nam-1")
    fitter.fit_audio(units, _synth_fixed(8.0), str(tmp_path), FCFG)
    assert units[0].tempo == 1.0 and units[0].strategy == "natural"


def test_min_fill_zero_disables(tmp_path):
    seg = Segment(id=0, start=0.0, end=10.0, text="x")
    units = fitter.build_units([seg], [10.0], {}, "nam-1")
    fitter.fit_audio(units, _synth_fixed(3.0), str(tmp_path), FCFG | {"min_fill": 0.0})
    assert units[0].tempo == 1.0 and units[0].tts_dur == pytest.approx(3.0, abs=0.02)
```

- [ ] **Step 2: RED** — AttributeError `tempo` / tts_dur 3.0.

- [ ] **Step 3: Code** — `pipeline/fitter.py`: thêm field `tempo: float = 1.0` vào `DubUnit`; helper:

```python
def _slow_down(path: str, tempo: float) -> None:
    tmp = path + ".slow.wav"
    subprocess.run([FFMPEG_EXE, "-y", "-v", "error", "-i", path, "-af", f"atempo={tempo:.4f}",
                    "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "1", tmp], check=True, capture_output=True)
    os.replace(tmp, path)
```

(`import subprocess`, `from .ffmpeg_utils import FFMPEG_EXE`). `_measure(units, fcfg=None)`: sau khi đo `tts_dur`, nếu `fcfg` và `min_fill > 0` và `budget > 0` và `tts_dur < min_fill * budget`: tính tempo, nếu `< 0.995` → `_slow_down`, đo lại, `unit.tempo = round(tempo, 3)`, `unit.strategy = "slowed" if unit.strategy == "natural" else unit.strategy + "+slowed"`; rồi mới tính `over_s`/`over` như cũ. `fit_audio` gọi `_measure(units, fcfg)`. Config: `fit.min_fill: 0.0  # >0: slow speech (atempo ≥ min_tempo, pitch kept) when a unit fills less than this share of its slot; presets 0.6`, `fit.min_tempo: 0.85`.

- [ ] **Step 4: GREEN + commit** `feat(fitter): gentle slow-down for under-filled units`.

---

### Task 22: Pipelining — dịch batch N+1 song song TTS batch N

**Files:** Modify `pipeline/fitter.py` (`run_pipelined`), `pipeline/orchestrator.py` (nhánh `fit.pipelined`), `config/default.yaml` (`fit.pipelined: false`), presets (`true`). Test: `tests/test_fitter_pipelined.py`, `tests/test_orchestrator_fit.py` (+1).

**Interfaces:**
- `fitter.run_pipelined(segments, slots, translate_fn, shorten_fn, synth, synth_batch, out_dir, fcfg, batch_size, workers) -> tuple[list[Segment], list[DubUnit]]` — `translate_fn(batch: list[Segment], budgets: list[tuple[float,int]]) -> list[Segment]`; trả `(translated_segments_in_id_order, units_in_id_order)`; mọi unit đã qua `fit_text` + `fit_audio` (có `_measure`).
- Cơ chế: chia `segments` thành batch liên tiếp; `ThreadPoolExecutor(max_workers=workers)` cho dịch (tối đa `workers` batch đang bay, nạp batch mới khi một batch xong); luồng chính tiêu thụ theo thứ tự hoàn thành: `build_units` → `fit_text` (shorten, gọi LLM) → `fit_audio(chunk, synth, out_dir, fcfg, synth_batch)`; lỗi trong pool được raise; kết quả gộp sắp theo id.
- Orchestrator: `if fit_enabled and fit_cfg.get("pipelined")`: bỏ qua `translate_segments`, gọi `run_pipelined` với `translate_fn` bọc `translate_segments(batch, target, client, source, tracker=, style_directives=, style_temperature=, budgets=)`, `workers = translation.parallel_batches`, `batch_size = translation.batch_size`; sau đó persist `translated`, `apply_units`, `save_units`, `fitted` như cũ.

- [ ] **Step 1: Test** — `tests/test_fitter_pipelined.py`:

```python
import threading, time
from dataclasses import replace
import numpy as np, soundfile as sf
from pipeline import fitter
from pipeline.transcriber import Segment

FCFG = dict(sec_per_syllable=0.2, overrun_tolerance=1.3, max_shorten_rounds=2, max_pause_borrow_s=0.6, margin_s=0.05, batch_chunk=32)


def _segs(n):
    return [Segment(id=i, start=i * 2.0, end=i * 2.0 + 1.5, text=f"src {i}") for i in range(n)]


class Probe:
    def __init__(self): self.lock = threading.Lock(); self.active = 0; self.max_active = 0; self.synth_calls = 0; self.overlap = False
    def translate(self, batch, budgets):
        with self.lock:
            self.active += 1; self.max_active = max(self.max_active, self.active)
        time.sleep(0.15)
        with self.lock: self.active -= 1
        return [replace(s, text=f"vi {s.id}", source_text=s.text) for s in batch]
    def synth_batch(self, texts, voice, paths):
        with self.lock:
            self.synth_calls += 1
            if self.active > 0: self.overlap = True
        time.sleep(0.1)
        for p in paths: sf.write(p, np.zeros(44100 // 2, dtype=np.float32), 44100)
        return list(paths)


def test_pipelined_overlaps_and_preserves_order(tmp_path):
    segs = _segs(8); slots = fitter.compute_slots(segs, 20.0, 0.6, 0.05)
    p = Probe()
    translated, units = fitter.run_pipelined(segs, slots, p.translate, lambda *a: "", None, p.synth_batch, str(tmp_path), FCFG, batch_size=2, workers=2)
    assert [s.id for s in translated] == list(range(8)) and [s.text for s in translated] == [f"vi {i}" for i in range(8)]
    assert [u.seg_id for u in units] == list(range(8)) and all(u.tts_dur > 0 for u in units)
    assert p.synth_calls == 4 and p.max_active >= 2 and p.overlap


def test_pipelined_matches_sequential(tmp_path):
    segs = _segs(5); slots = fitter.compute_slots(segs, 12.0, 0.6, 0.05)
    p = Probe()
    translated, units = fitter.run_pipelined(segs, slots, p.translate, lambda *a: "", None, p.synth_batch, str(tmp_path / "p"), FCFG, batch_size=2, workers=1)
    seq = p.translate(segs, [])
    u2 = fitter.build_units(seq, slots, {}, "nam-1"); fitter.fit_text(u2, lambda *a: "", FCFG)
    (tmp_path / "s").mkdir(); fitter.fit_audio(u2, None, str(tmp_path / "s"), FCFG, synth_batch=p.synth_batch)
    assert [(u.seg_id, u.text, round(u.tts_dur, 2)) for u in units] == [(u.seg_id, u.text, round(u.tts_dur, 2)) for u in u2]


def test_pipelined_propagates_translate_errors(tmp_path):
    segs = _segs(4); slots = fitter.compute_slots(segs, 10.0, 0.6, 0.05)
    def boom(batch, budgets): raise RuntimeError("llm down")
    import pytest
    with pytest.raises(RuntimeError, match="llm down"):
        fitter.run_pipelined(segs, slots, boom, lambda *a: "", None, Probe().synth_batch, str(tmp_path), FCFG, batch_size=2, workers=2)
```

`tests/test_orchestrator_fit.py`: thêm test giống test hiện có nhưng `monkeypatch.setitem(cfg["fit"], "pipelined", True)` và patch `pipeline.orchestrator.fitter.run_pipelined` trả `(translated, units)` giả → assert `translate_segments` KHÔNG được gọi trực tiếp và artifacts vẫn ghi.

- [ ] **Step 2: RED** — AttributeError `run_pipelined`.

- [ ] **Step 3: Code** — `pipeline/fitter.py`:

```python
def run_pipelined(segments, slots, translate_fn, shorten_fn, synth, synth_batch, out_dir, fcfg, batch_size, workers=2):
    """Translate batch N+1 while batch N is being shortened and synthesized.

    The translation pool keeps at most `workers` LLM batches in flight; the
    calling thread consumes finished batches (completion order), runs the
    LLM shortening pass and TTS for that batch, and results are reassembled
    by segment id. Errors from any batch propagate.
    """
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    voice_map = fcfg.get("_voice_map") or {}
    default_voice = fcfg.get("_default_voice", "")
    batches = [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]
    slot_by_id = {s.id: sl for s, sl in zip(segments, slots)}
    sps = float(fcfg.get("sec_per_syllable", 0.21))
    translated_all: dict[int, Segment] = {}
    units_all: dict[int, DubUnit] = {}
    os.makedirs(out_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending, next_i = set(), 0
        while next_i < len(batches) and len(pending) < max(1, workers):
            b = batches[next_i]; pending.add(pool.submit(translate_fn, b, budgets_for(b, [slot_by_id[s.id] for s in b], sps))); next_i += 1
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            while next_i < len(batches) and len(pending) < max(1, workers):
                b = batches[next_i]; pending.add(pool.submit(translate_fn, b, budgets_for(b, [slot_by_id[s.id] for s in b], sps))); next_i += 1
            for fut in done:
                tr = fut.result()                       # raises translation errors here
                for s in tr: translated_all[s.id] = s
                units = build_units(tr, [slot_by_id[s.id] for s in tr], voice_map, default_voice)
                fit_text(units, shorten_fn, fcfg)
                fit_audio(units, synth, out_dir, fcfg, synth_batch=synth_batch)
                for u in units: units_all[u.seg_id] = u
                print(f"      [pipeline] {len(units_all)}/{len(segments)} units done")
    ids = sorted(units_all)
    return [translated_all[i] for i in ids], [units_all[i] for i in ids]
```

`voice_map`/`default_voice` được truyền qua `fcfg` dưới khoá `_voice_map`/`_default_voice` (orchestrator gán `fit_cfg = dict(fit_cfg, _voice_map=voice_map, _default_voice=effective_voice)`) để không đổi chữ ký các hàm M1. `pipeline/orchestrator.py`: trong nhánh fit, nếu `fit_cfg.get("pipelined")`:

```python
            def _translate_batch(batch, budgets):
                return translate_segments(batch, opts.target_language, translation_client, opts.source_language,
                                          tracker=tracker, style_directives=style.translation_directives,
                                          style_temperature=style.temperature, budgets=budgets)
            translated, units = fitter.run_pipelined(
                segments, slots, _translate_batch, _shorten, synth, synth_batch, str(tts_dir),
                dict(fit_cfg, _voice_map=voice_map, _default_voice=effective_voice),
                batch_size=int(cfg["translation"].get("batch_size", 32)),
                workers=int(cfg["translation"].get("parallel_batches", 1)),
            )
            _persist_segments(translated, output_video_path, "translated")
```

(bỏ lời gọi `translate_segments`/`fit_text`/`fit_audio` tuần tự trong nhánh này; `_shorten`, `synth`, `synth_batch`, `voice_map`, `effective_voice` được tạo trước; `voice_map = {}` cho tới Task 24). Thứ tự `_emit` step 3/4 giữ hợp lý ("Translating + synthesizing (pipelined)…"). Config: `fit.pipelined: false  # translate batch N+1 while TTS batch N runs (two GPUs / GPU + CPU)`; presets `true`.

- [ ] **Step 4: GREEN + commit** `perf(fitter): pipeline translation and TTS batches across devices`.

---

### Task 23: Diarization theo câu — `pipeline/diarizer.py`

**Files:** Create `pipeline/diarizer.py`; Modify `config/default.yaml` (khối `diarization`), `pyproject.toml` (`local-gpu` + `speechbrain>=1.0`, `scipy>=1.11`; `uv lock`). Test: `tests/test_diarizer.py`.

**Interfaces:**
- `Turn(start: float, end: float, speaker: str)`.
- `label_segments(audio_path: str, segments: list[Segment], *, backend="ecapa", num_speakers: int | None = None, max_speakers: int = 4, threshold: float = 0.65, hf_token: str | None = None, model: str = "speechbrain/spkrec-ecapa-voxceleb", pyannote_model: str = "pyannote/speaker-diarization-community-1", device="auto") -> list[str]` — nhãn `SPEAKER_00..` theo thứ tự xuất hiện, 1 nhãn/segment.
- Pure helpers (test không cần model): `cluster_embeddings(embs: np.ndarray, num_speakers=None, max_speakers=4, threshold=0.65) -> list[int]` (agglomerative, cosine, average linkage; `num_speakers` → `fcluster(maxclust)`, else `fcluster(t=threshold, criterion="distance")` rồi cắt về `max_speakers` bằng maxclust nếu vượt); `relabel_by_first_appearance(labels: list[int]) -> list[str]`; `assign_by_overlap(segments, turns: list[Turn]) -> list[str]` (speaker có tổng overlap lớn nhất; không overlap → speaker của turn gần nhất).
- Backend `ecapa`: đọc WAV 16 kHz (`soundfile`), với mỗi segment cắt `[start, end]` (tối thiểu 0.5 s, mở rộng đối xứng nếu ngắn), `EncoderClassifier.from_hparams(source=model, run_opts={"device": device})`.encode_batch → embedding L2-normalized; cluster → relabel. Segment < 0.3 s hoặc lỗi → nhãn của segment liền trước.
- Backend `pyannote`: `Pipeline.from_pretrained(pyannote_model, token=hf_token)` (lazy import) → `assign_by_overlap`.
- Config: `diarization: {enabled: false, backend: ecapa, num_speakers: null, max_speakers: 4, threshold: 0.65, model: speechbrain/spkrec-ecapa-voxceleb, pyannote_model: pyannote/speaker-diarization-community-1, device: auto}`.

- [ ] **Step 1: Test** — `tests/test_diarizer.py`:

```python
import numpy as np, pytest
from pipeline import diarizer
from pipeline.diarizer import Turn
from pipeline.transcriber import Segment


def _embs(centers, n_each=5, noise=0.02, seed=0):
    rng = np.random.default_rng(seed); out = []
    for c in centers:
        for _ in range(n_each):
            v = c + rng.normal(0, noise, size=c.shape); out.append(v / np.linalg.norm(v))
    return np.array(out)


def test_cluster_two_speakers_by_threshold():
    a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    labels = diarizer.cluster_embeddings(_embs([a, b]), threshold=0.5)
    assert len(set(labels[:5])) == 1 and len(set(labels[5:])) == 1 and labels[0] != labels[5]


def test_cluster_fixed_num_speakers_and_cap():
    a, b, c = np.eye(3)
    labels = diarizer.cluster_embeddings(_embs([a, b, c]), num_speakers=2)
    assert len(set(labels)) == 2
    labels = diarizer.cluster_embeddings(_embs([a, b, c]), threshold=0.1, max_speakers=2)
    assert len(set(labels)) <= 2


def test_relabel_by_first_appearance():
    assert diarizer.relabel_by_first_appearance([7, 7, 2, 7, 2, 9]) == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]


def test_assign_by_overlap_and_nearest():
    turns = [Turn(0.0, 5.0, "A"), Turn(5.0, 10.0, "B")]
    segs = [Segment(id=0, start=1.0, end=4.0, text="x"), Segment(id=1, start=4.5, end=7.0, text="y"), Segment(id=2, start=12.0, end=13.0, text="z")]
    assert diarizer.assign_by_overlap(segs, turns) == ["A", "B", "B"]


def test_label_segments_ecapa_with_fake_encoder(tmp_path, monkeypatch):
    import soundfile as sf
    sr = 16000; sf.write(tmp_path / "a.wav", np.zeros(sr * 12, dtype=np.float32), sr)
    segs = [Segment(id=i, start=i * 2.0, end=i * 2.0 + 1.5, text="s") for i in range(6)]
    class FakeEnc:
        def encode(self, wav, sr):   # returns embedding by time: first 3 segs → e1, rest → e2
            t = getattr(wav, "_t", 0.0); return np.array([1.0, 0.0]) if t < 6.0 else np.array([0.0, 1.0])
    def fake_loader(model, device):
        enc = FakeEnc()
        def embed(wav, sr, t0):
            wav._t = t0; return enc.encode(wav, sr)
        return embed
    monkeypatch.setattr(diarizer, "_load_ecapa_embedder", fake_loader)
    labels = diarizer.label_segments(str(tmp_path / "a.wav"), segs, backend="ecapa", threshold=0.5)
    assert labels == ["SPEAKER_00"] * 3 + ["SPEAKER_01"] * 3
```

(`_load_ecapa_embedder(model, device) -> Callable[[np.ndarray, int, float], np.ndarray]` — hàm thật trả closure gọi speechbrain; test thay bằng closure giả nhận `(wav_crop, sr, t0)`. Trong `label_segments`, `wav_crop` là mảng numpy cắt từ file; test gắn thuộc tính `_t` lên mảng — numpy array không cho gán thuộc tính → dùng subclass: implement `label_segments` truyền `t0=seg.start` làm tham số thứ 3 và test dùng `t0`, bỏ `_t`. Viết test theo chữ ký `embed(wav, sr, t0)`.)

- [ ] **Step 2: RED** — ModuleNotFoundError.

- [ ] **Step 3: Code** — `pipeline/diarizer.py` với các hàm trên; `_load_ecapa_embedder`:

```python
def _load_ecapa_embedder(model: str, device: str):
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from .devices import pick_device
    dev = pick_device(device)
    clf = EncoderClassifier.from_hparams(source=model, run_opts={"device": dev})
    def embed(wav: np.ndarray, sr: int, t0: float) -> np.ndarray:
        x = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0).to(dev)
        with torch.no_grad():
            e = clf.encode_batch(x).squeeze().detach().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)
    return embed
```

`label_segments` (ecapa): đọc audio bằng `soundfile.read(audio_path, dtype="float32")` (mono, 16 kHz — file `audio.wav` của `extract_audio`); crop có mở rộng tới ≥ 0.5 s; gom embeddings; `cluster_embeddings`; `relabel_by_first_appearance`. Clustering dùng `scipy.cluster.hierarchy.linkage(embs, method="average", metric="cosine")` + `fcluster`. Config block + extras (`speechbrain>=1.0`, `scipy>=1.11` trong `local-gpu`).

- [ ] **Step 4: GREEN + commit** `feat(diarizer): sentence-level speaker labelling (ECAPA clustering, optional pyannote)`.

---

### Task 24: Gán giọng theo speaker, giới tính theo F0, cờ CLI, nối orchestrator

**Files:** Modify `pipeline/voices.py` (`speaker_voices` round-robin, `guess_genders`), `pipeline/orchestrator.py` (`DubOptions.speakers`, `voice_map`; gọi diarizer sau merge/split; `voice_map` vào `build_units`/`run_pipelined`/`synthesize_segments`; persist `diarized` + `voices.json`), `main.py` (`--speakers {1,auto,N}`, `--voice-map`), `config/default.yaml` (`voices.speaker_voices`, `voices.gender_detect: false`), presets (`gender_detect: true` trên GPU). Test: `tests/test_voices_speakers.py`, `tests/test_orchestrator_speakers.py`.

**Interfaces:**
- `voices.assign_voices(speakers, default_voice, voice_map=None, genders=None, bank=None, speaker_voices=None)` — thêm tham số `speaker_voices: list[str] | None`: khi speaker không có trong `voice_map` và không biết giới tính → lấy round-robin từ `speaker_voices` (giữ hành vi cũ khi `speaker_voices` None); khi biết giới tính → giọng nam/nữ đầu tiên trong `speaker_voices` có giới tính khớp (catalog VieNeu: `voices.preset_genders` map tên → gender) hoặc `default_male/female`.
- `voices.guess_genders(audio_path, segments) -> dict[str, str]` — với mỗi speaker, ghép tối đa 12 s audio 16 kHz từ các segment của họ, tính F0 trung vị bằng autocorrelation numpy trên khung 40 ms (voiced khi năng lượng > ngưỡng, F0 trong 70–400 Hz); `< 165 Hz` → `male`, ngược lại `female`; không đủ khung voiced → không trả speaker đó.
- `DubOptions.speakers: str = "1"` (`"1"` tắt diarization, `"auto"` tự đếm, `"N"` số cố định), `DubOptions.voice_map: dict[str, str] | None`.
- Orchestrator: sau merge + split_long: nếu `speakers != "1"` (hoặc `diarization.enabled`): `labels = diarizer.label_segments(audio_path, segments, backend=…, num_speakers=int(N) hoặc None, …)`; gán `seg.speaker`; `raw_sentences` cũng gán theo overlap với segments (để subtitle giữ speaker); persist `diarized`; `genders = guess_genders(...)` nếu `voices.gender_detect`; `voice_map = assign_voices(sorted speakers by first appearance, effective_voice, opts.voice_map, genders, speaker_voices=cfg voices.speaker_voices)`; ghi `<output>.voices.json`; truyền `voice_map` vào `build_units`/`run_pipelined`/`synthesize_segments`. Lưu ý `merge_continuous_segments` không gộp khác speaker → diarization phải chạy TRƯỚC merge để tránh gộp 2 người: thứ tự: transcribe → (diarize + gán speaker) → merge → split_long. Yêu cầu `audio_path` 16 kHz — trên nhánh `segments_override` (captions) vẫn extract audio khi cần diarize.
- CLI: `--speakers auto`, `--voice-map "SPEAKER_00=Phạm Tuyên,SPEAKER_01=Ngọc Huyền"`.
- **Quyết định user (2026-09-02):** giọng nam mặc định = `Thanh Bình` (đã nghe thử 10 preset). Đổi `voices.default_male: "Thanh Bình"` trong `config/default.yaml` và 2 preset; `fit.sec_per_syllable: 0.22` trong 2 preset (đo bằng `scripts/calibrate_voice.py`: Thanh Bình 0.217, Ngọc Huyền 0.230, Minh Đức 0.270).
- Config: `voices.speaker_voices: ["Thanh Bình", "Ngọc Huyền", "Minh Đức", "Trúc Ly", "Thái Sơn", "Mai Anh"]`, `voices.preset_genders: {"Phạm Tuyên": male, "Ngọc Huyền": female, "Minh Đức": male, "Trúc Ly": female, "Thái Sơn": male, "Mai Anh": female, "Adam": male, "Quang Sơn": male, "Ngọc Trân": female, "Xuân Vĩnh": male, "Minh Triết": male, "Đức Trí": male, "Thục Đoan": female, "Thùy Dung": female, "Mỹ Duyên": female, "Kim Thanh": female, "Thanh Bình": male, "Ngọc Linh": female, "Đoan Trang": female, "Quỳnh Anh": female}`, `voices.gender_detect: false`.

- [ ] **Step 1: Test** — `tests/test_voices_speakers.py`: round-robin theo thứ tự xuất hiện; giới tính → giọng khớp; `voice_map` thắng; `guess_genders` trên WAV tổng hợp: speaker A = sóng răng cưa 120 Hz (→ male), B = 220 Hz (→ female), có đoạn im lặng. `tests/test_orchestrator_speakers.py`: patch `diarizer.label_segments` trả `["SPEAKER_00","SPEAKER_01"]`, `guess_genders` trả `{}`; `DubOptions(speakers="auto")` → `synthesize_segments`/`build_units` nhận `voice_map` với 2 giọng khác nhau; `voices.json` được ghi; với `speakers="1"` không gọi diarizer.

- [ ] **Step 2: RED.** — [ ] **Step 3: Code** theo Interfaces (F0: khung 640 mẫu, hop 320; autocorr chuẩn hoá; đỉnh trong lag 40–228 mẫu; voiced khi RMS > 0.01 và đỉnh > 0.5). — [ ] **Step 4: GREEN + commit** `feat(speakers): per-speaker preset voices, F0 gender guess, --speakers/--voice-map`.

---

### Task 25: E2E M2 trên server + docs

- [ ] **Step 1:** rsync; `uv pip install speechbrain scipy` trong `/workspace/violin/.venv` (đã cài); chạy lại clip DeepMind 21 phút với preset GPU mới (`pipelined`, `min_fill`, `max_sentence_seconds`) — đo tổng thời gian (kỳ vọng ≤ 4 phút), kiểm tra `fit.units.json`: `max unit span ≤ 12 s`, tỉ lệ `slowed`, fill.
- [ ] **Step 2:** clip phỏng vấn 2 người (CC, 3–10 phút, tìm bằng yt-dlp `--match-filter "license~=(?i)creative commons"`): chạy `--speakers auto`; kiểm tra `diarized.segments.json` có 2 speaker, `voices.json` 2 giọng khác nhau, nghe 3 đoạn.
- [ ] **Step 3:** README: mục Multi-speaker (`--speakers auto`, `--voice-map`, cần `speechbrain` trên GPU) + `fit.pipelined`, `fit.min_fill`, `transcription.max_sentence_seconds`; spec Amendment M2. Commit `docs: M2 multi-speaker and pipelining guide`.

## Self-review

- Coverage: (1) pipelining Task 22; (2) diarization + voices Tasks 23–24; (3) split Task 20; (4) slow-down Task 21; E2E Task 25.
- Type consistency: `run_pipelined` dùng `translate_fn(batch, budgets)`; `label_segments` trả list[str] cùng độ dài segments; `assign_voices` thêm tham số keyword mới, tương thích Task 5.
- Ordering: diarize trước merge (Task 24) — Task 20's split chạy sau merge và giữ `speaker`.
