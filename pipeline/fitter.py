"""Duration fitter: make each Vietnamese sentence fit its time slot.

Phase A (`fit_text`, LLM only):  estimate seconds from syllables; when the
    estimate exceeds the slot budget by more than `overrun_tolerance`, ask the
    LLM to shorten the translation (≤ `max_shorten_rounds`).
Phase B (`fit_audio`, TTS only): synthesize each unit once at natural speed,
    measure, and flag units still longer than their budget (`over_s`); the
    TTS has no speed control, so the merger absorbs the residual.
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
    strategy: str = "natural"   # natural | shortened | over | shortened+over
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


def _measure(units: list[DubUnit]) -> None:
    """Read each unit's `tts_path` duration and flag units over their budget (`over_s`)."""
    over = 0
    for i, unit in enumerate(units):
        unit.tts_dur = wav_duration(unit.tts_path)
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
    `fcfg` is accepted for interface symmetry with `fit_text`.

    When `synth_batch` is given, units are grouped by voice (first-appearance
    order) and synthesized one `synth_batch` call per group (GPU static
    batching); otherwise each unit is synthesized one at a time via `synth`.
    Either way, measurement (`tts_dur`/`over_s`/`strategy`) is identical.
    """
    _ = fcfg
    os.makedirs(out_dir, exist_ok=True)
    if synth_batch is not None:
        groups: dict[str, list[DubUnit]] = {}
        for unit in units:
            groups.setdefault(unit.voice, []).append(unit)
        for voice, group in groups.items():
            paths = [str(Path(out_dir) / f"seg_{u.seg_id:05d}.wav") for u in group]
            out_paths = synth_batch([u.text for u in group], voice, paths)
            for unit, path in zip(group, out_paths):
                unit.tts_path = path
            print(f"      [fit] phase B: batch {voice} {len(group)} units")
    else:
        for unit in units:
            path = str(Path(out_dir) / f"seg_{unit.seg_id:05d}.wav")
            unit.tts_path = synth(unit.text, unit.voice, path, 1.0)
    _measure(units)


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
