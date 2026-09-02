"""Two-pass TTS for natural voice + per-segment alignment.

Pass 1: synth every segment at speed=1.0 (natural prosody), measure duration.
Pass 2: re-synth ONLY segments where pass-1 audio overran the source slot
        (allowing for video stretch up to 1/clamp_min), at a per-segment
        speed capped at --max-speed. Slack segments stay at 1.0.

The merger's speed_clamp_min still absorbs residual overflow, so the
target slot here is `slot * 1/clamp_min` (segments need to fit that, not
the raw source slot).

Usage:
    uv run two_pass_tts.py \\
        --segments /tmp/violin_mit15/output_vi_A.translated.segments.json \\
        --tts-dir /tmp/violin_mit15/tts_B \\
        --language Vietnamese \\
        --config config/local_mac.yaml
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import wave
from pathlib import Path

from pipeline import config as pipeline_config
from pipeline.costs import CostTracker
from pipeline.languages import language_code
from pipeline.transcriber import Segment, merge_continuous_segments, split_into_sentences
from pipeline.tts_supertonic import (
    get_shared_tts,
    native_voices_for,
    synthesize_segment,
)


def _wav_dur(path: str) -> float:
    with contextlib.closing(wave.open(path, "r")) as f:
        return f.getnframes() / float(f.getframerate())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segments", required=True, help="Path to *.translated.segments.json")
    p.add_argument("--tts-dir", required=True, help="Output dir for TTS WAVs")
    p.add_argument("--language", default="Vietnamese")
    p.add_argument("--config", default="config/local_mac.yaml")
    p.add_argument("--voice", default=None, help="Override voice (default: native by gender)")
    p.add_argument("--gender", default="male", choices=["male", "female"])
    p.add_argument("--pass1-speed", type=float, default=1.0, help="Speed for pass 1 (natural=1.0)")
    p.add_argument("--max-speed", type=float, default=1.4, help="Max per-segment speed in pass 2")
    args = p.parse_args()

    pipeline_config.load(args.config)
    cfg = pipeline_config.get()
    clamp_min = float(cfg["merge_video"].get("speed_clamp_min", 1.0))
    target_slot_factor = 1.0 / clamp_min if clamp_min > 0 else 1.0

    data = json.loads(Path(args.segments).read_text(encoding="utf-8"))
    segs_raw = [
        Segment(
            id=s["id"], start=float(s["start"]), end=float(s["end"]),
            text=s["text"], speaker=s.get("speaker", "SPEAKER_00"),
        )
        for s in data["segments"]
    ]
    # Reproduce orchestrator's post-translate transform.
    segs = merge_continuous_segments(segs_raw, max_duration=float("inf"))
    segs = split_into_sentences(segs)
    print(f"[two-pass] {len(segs)} segments after merge+split (from {len(segs_raw)} raw)")

    lang = language_code(args.language)
    if args.voice:
        voice = args.voice
    else:
        male, female = native_voices_for(lang)
        voice = male if args.gender == "male" else female
    print(f"[two-pass] voice={voice} language={lang} pass1_speed={args.pass1_speed} max_speed={args.max_speed} clamp_min={clamp_min}")

    tts = get_shared_tts()
    out_dir = Path(args.tts_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = CostTracker()

    # ── Pass 1 ────────────────────────────────────────────────────────
    print(f"[pass 1] synth {len(segs)} segments at speed={args.pass1_speed}")
    pass1_dur: list[float] = []
    for i, seg in enumerate(segs):
        path = str(out_dir / f"seg_{seg.id:05d}.wav")
        synthesize_segment(seg.text, voice, path, tts, lang, speed=args.pass1_speed)
        pass1_dur.append(_wav_dur(path))
        tracker.add_tts_usage(len(seg.text))
        if (i + 1) % 20 == 0 or (i + 1) == len(segs):
            print(f"   pass1 {i + 1}/{len(segs)}")

    # ── Compute per-segment target speed ──────────────────────────────
    speeds: list[float] = []
    rerun = 0
    for seg, d in zip(segs, pass1_dur):
        slot = seg.end - seg.start
        if slot < 0.01:
            speeds.append(1.0); continue
        target = slot * target_slot_factor  # what merger can absorb
        if d > target:
            needed = d / target
            sp = min(args.max_speed, needed)
            sp = round(sp, 2)
        else:
            sp = 1.0
        speeds.append(sp)
        if sp > 1.0:
            rerun += 1

    # Distribution
    buckets: dict[float, int] = {}
    for sp in speeds:
        b = round(sp, 1)
        buckets[b] = buckets.get(b, 0) + 1
    print(f"[pass 2] {rerun}/{len(segs)} segments need speed > 1.0")
    print(f"   distribution: {dict(sorted(buckets.items()))}")
    at_cap = sum(1 for sp in speeds if sp >= args.max_speed - 0.001)
    print(f"   {at_cap} segments capped at max_speed={args.max_speed} (will leak into video stretch)")

    # ── Pass 2 ────────────────────────────────────────────────────────
    done = 0
    for i, (seg, sp) in enumerate(zip(segs, speeds)):
        if sp <= 1.0:
            continue
        path = str(out_dir / f"seg_{seg.id:05d}.wav")
        synthesize_segment(seg.text, voice, path, tts, lang, speed=sp)
        tracker.add_tts_usage(len(seg.text))
        done += 1
        if done % 20 == 0 or done == rerun:
            print(f"   pass2 {done}/{rerun}")

    print(f"\n[two-pass] done. TTS WAVs at: {out_dir}")
    print(f"   TTS chars total (pass1+pass2): {tracker.tts_characters}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
