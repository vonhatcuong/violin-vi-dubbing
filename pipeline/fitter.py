"""Duration fitter: make each Vietnamese sentence fit its time slot.

Phase A (`fit_text`, LLM only):  estimate seconds from syllables; when the
    estimate exceeds the slot budget by more than `overrun_tolerance`, ask the
    LLM to shorten the translation (≤ `max_shorten_rounds`).
Phase B (`fit_audio`, TTS only): synthesize each unit once at natural speed,
    measure, and flag units still longer than their budget (`over_s`); the
    TTS has no speed control, so the merger absorbs the residual. Units that
    fill less than `fit.min_fill` of their slot are gently slowed instead
    (ffmpeg `atempo` ≥ `fit.min_tempo`, pitch kept; off by default).
`apply_units` then extends `Segment.end` to borrow the following pause
(bounded by `slot_end`); whatever is still over is absorbed by the merger
(video slow-down ≤ 8 %, atempo ≤ 1.4, hard trim).

Units are persisted as `<output>.fit.units.json` for inspection.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import soundfile as sf

from .ffmpeg_utils import FFMPEG_EXE
from .transcriber import Segment
from .vi_text import count_syllables

ShortenFn = Callable[[str, str, int, float], str]
SynthFn = Callable[[str, str, str, float], str]
BatchSynthFn = Callable[[list[str], str, list[str]], list[str]]


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
    strategy: str = "natural"   # natural | shortened | over | shortened+over | slowed | ...+slowed
    rounds: int = 0
    over_s: float = 0.0          # seconds still over budget after phase B (merger absorbs)
    tempo: float = 1.0           # ffmpeg atempo applied to under-filled units (≤ 1; 1 = unchanged)

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
        out.append((seconds, max(1, int(seconds / sec_per_syllable + 1e-6))))
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
        budget_syll = max(1, int(budget / sps + 1e-6))
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


def _slow_down(path: str, tempo: float) -> None:
    """Rewrite the WAV at `path` in place, slowed by ffmpeg `atempo` (44.1 kHz mono PCM16)."""
    tmp = path + ".slow.wav"
    subprocess.run([FFMPEG_EXE, "-y", "-v", "error", "-i", path, "-af", f"atempo={tempo:.4f}",
                    "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "1", tmp], check=True, capture_output=True)
    os.replace(tmp, path)


def _measure(units: list[DubUnit], fcfg: dict | None = None) -> None:
    """Read each unit's `tts_path` duration and flag units over their budget (`over_s`).

    When `fcfg["min_fill"] > 0`, a unit whose speech fills less than that
    share of its slot is gently slowed (ffmpeg `atempo`, pitch kept, never
    below `fcfg["min_tempo"]`) before the overrun is computed.
    """
    min_fill = float((fcfg or {}).get("min_fill", 0.0))
    min_tempo = float((fcfg or {}).get("min_tempo", 0.85))
    over = 0
    for i, unit in enumerate(units):
        unit.tts_dur = wav_duration(unit.tts_path)
        budget = unit.budget_s
        if min_fill > 0 and budget > 0 and unit.tts_dur < min_fill * budget:
            tempo = max(min_tempo, unit.tts_dur / (min_fill * budget))
            if tempo < 0.995:
                _slow_down(unit.tts_path, tempo)
                unit.tts_dur = wav_duration(unit.tts_path)
                unit.tempo = round(tempo, 3)
                unit.strategy = "slowed" if unit.strategy == "natural" else unit.strategy + "+slowed"
        unit.over_s = round(max(0.0, unit.tts_dur - unit.budget_s), 3)
        if unit.over_s > 0:
            unit.strategy = "shortened+over" if unit.rounds else "over"
            over += 1
        if (i + 1) % 10 == 0 or i + 1 == len(units):
            print(f"      [fit] phase B: {i + 1}/{len(units)} synthesized ({over} over budget)")


def fit_audio(
    units: list[DubUnit], synth: SynthFn, out_dir: str, fcfg: dict, synth_batch: BatchSynthFn | None = None,
) -> None:
    """Phase B: synthesize each unit once at natural speed and measure it.

    VieNeu has no speed control, so nothing is re-synthesized here; units
    still longer than their budget are flagged (`over_s`) and the merger
    absorbs the overrun (video slow-down ≤ 8 %, atempo ≤ 1.4, hard trim).
    `fcfg` is passed through to `_measure`, which uses `min_fill`/`min_tempo`
    to gently slow (ffmpeg `atempo`) units that under-fill their slot.

    When `synth_batch` is given, units are grouped by voice (first-appearance
    order), split into chunks of at most `fcfg["batch_chunk"]` units, and
    synthesized one `synth_batch` call per chunk (GPU static batching);
    otherwise each unit is synthesized one at a time via `synth`. Either way,
    measurement (`tts_dur`/`over_s`/`strategy`) is identical.
    """
    os.makedirs(out_dir, exist_ok=True)
    if synth_batch is not None:
        batch_chunk = int(fcfg.get("batch_chunk", 32))
        groups: dict[str, list[DubUnit]] = {}
        for unit in units:
            groups.setdefault(unit.voice, []).append(unit)
        done = 0
        for voice, group in groups.items():
            for start in range(0, len(group), batch_chunk):
                chunk = group[start:start + batch_chunk]
                paths = [str(Path(out_dir) / f"seg_{u.seg_id:05d}.wav") for u in chunk]
                out_paths = synth_batch([u.text for u in chunk], voice, paths)
                if len(out_paths) != len(chunk):
                    raise RuntimeError(
                        f"synth_batch returned {len(out_paths)} paths for {len(chunk)} texts (voice={voice!r})"
                    )
                for unit, path in zip(chunk, out_paths):
                    if not os.path.exists(path):
                        raise RuntimeError(f"synth_batch did not write expected output {path!r} (voice={voice!r})")
                    unit.tts_path = path
                done += len(chunk)
                print(f"      [fit] phase B: {done}/{len(units)} synthesized (batch {voice})")
    else:
        for unit in units:
            path = str(Path(out_dir) / f"seg_{unit.seg_id:05d}.wav")
            unit.tts_path = synth(unit.text, unit.voice, path, 1.0)
    _measure(units, fcfg)


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


def run_pipelined(
    segments: list[Segment],
    slots: list[float],
    translate_fn: Callable[[list[Segment], list[tuple[float, int]]], list[Segment]],
    shorten_fn: ShortenFn,
    synth: SynthFn | None,
    synth_batch: BatchSynthFn | None,
    out_dir: str,
    fcfg: dict,
    batch_size: int,
    workers: int = 2,
) -> tuple[list[Segment], list[DubUnit]]:
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


def save_units(units: list[DubUnit], path: str | Path) -> None:
    payload = {"count": len(units), "units": [asdict(u) | {"budget_s": round(u.budget_s, 3)} for u in units]}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      [fit] units → {path}")
