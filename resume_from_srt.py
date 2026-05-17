"""Resume Violin pipeline from a previously-generated SRT.

Skips Extract + Transcribe + Translate (uses Vietnamese text and timestamps
already in the SRT) and only runs TTS + Merge. Useful when tuning Supertonic
speed or video alignment knobs — saves ~8 minutes per re-run.

Usage:
    uv run resume_from_srt.py \\
        --input  /path/to/source_video.mp4 \\
        --srt    /tmp/violin_out_v1/dubbed.srt \\
        --output /tmp/violin_out/dubbed_v2.mp4 \\
        --language Vietnamese \\
        --config config/local_mac.yaml
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import threading
from pathlib import Path

from pipeline import config as pipeline_config
from pipeline.costs import CostTracker
from pipeline.extractor import get_video_duration
from pipeline.languages import language_code
from pipeline.merger import build_aligned_video, build_gap_chunks, generate_srt, prepare_merge
from pipeline.styles import resolve as resolve_style
from pipeline.transcriber import Segment
from pipeline.tts import native_voices_for, synthesize_segments


_TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _ts_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path: str) -> list[Segment]:
    """Parse SRT file → list[Segment] with start/end (seconds) and text."""
    raw = Path(path).read_text(encoding="utf-8")
    # Normalize line endings, split on blank lines.
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]

    segments: list[Segment] = []
    for block in blocks:
        lines = block.split("\n")
        if len(lines) < 2:
            continue
        # First line is index; second is timestamp; rest is text.
        ts_match = _TS_RE.search(lines[1] if len(lines) >= 2 else "")
        if not ts_match:
            # Some SRT writers omit the index line — try line 0 for the timestamp.
            ts_match = _TS_RE.search(lines[0])
            text_lines = lines[1:]
        else:
            text_lines = lines[2:]
        if not ts_match:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = ts_match.groups()
        start = _ts_to_seconds(h1, m1, s1, ms1)
        end = _ts_to_seconds(h2, m2, s2, ms2)
        text = " ".join(text_lines).strip()
        if not text:
            continue
        segments.append(Segment(
            id=len(segments),
            start=start,
            end=end,
            text=text,
        ))
    return segments


def _resolve_voice(voice: str | None, lang_code: str, cfg: dict) -> str:
    if voice:
        return voice
    male, female = native_voices_for(lang_code)
    gender = cfg.get("preferences", {}).get("voice_gender", "male").lower()
    return male if gender == "male" else female


def _voiceover_volumes(voiceover: bool, cfg: dict) -> tuple[float, float, float]:
    """Match orchestrator._voiceover_volumes (CLI bake mode).

    Returns (mix_volume, original_audio_volume, gap_volume) in the exact order
    prepare_merge() expects. mix_volume is the volume at which the original
    audio is BAKED INTO each speech chunk (must be small or the dub gets buried).
    """
    if not voiceover:
        return 0.0, 0.0, 1.0
    vo_volume = float(cfg["merge_video"].get("voiceover_volume", 0.1))
    gap_vol = min(1.0, 2 * vo_volume)
    return vo_volume, 0.0, gap_vol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Source video file (for video frames + duration)")
    parser.add_argument("--srt", required=True, help="Previously-generated SRT with target-language text")
    parser.add_argument("--output", required=True, help="Output dubbed video path (.mp4)")
    parser.add_argument("--language", "-l", required=True, help="Target language name (e.g. Vietnamese)")
    parser.add_argument("--voice", "-v", default=None, help="TTS voice override (default: native)")
    parser.add_argument("--config", "-c", default="config/local_mac.yaml", help="Pipeline config YAML")
    parser.add_argument("--no-voiceover", action="store_true", help="Full replacement (no original audio underneath)")
    parser.add_argument("--no-subtitles", action="store_true", help="Skip writing a new SRT")
    parser.add_argument("--tts-dir", default=None, help="Reuse TTS WAVs from an existing directory (skip TTS step). Files must be named seg_NNNNN.wav matching segment IDs.")
    parser.add_argument("--keep-tts-dir", default=None, help="Path to keep TTS WAVs after run (for reuse via --tts-dir on later tune-only runs).")
    args = parser.parse_args()

    pipeline_config.load(args.config)
    cfg = pipeline_config.get()

    print(f"[resume] Parsing SRT → segments…")
    segments = parse_srt(args.srt)
    print(f"[resume]   {len(segments)} segments loaded from {args.srt}")
    if not segments:
        print("[resume] No segments parsed from SRT — abort.", file=sys.stderr)
        return 1

    total_duration = get_video_duration(args.input)
    print(f"[resume]   source video duration: {total_duration:.1f}s")

    # SRT timestamps from a previous Violin run are in OUTPUT-video space
    # (post-align, often longer than the source when target dub is slower).
    # The merger expects timestamps in INPUT-video space, so linearly rescale
    # when the SRT max-end overshoots the source duration. Preserves relative
    # timing across segments while fitting them back into the 0..total_duration
    # window.
    srt_span = max(seg.end for seg in segments)
    if srt_span > total_duration + 0.5:
        scale = total_duration / srt_span
        print(f"[resume]   rescaling timestamps {srt_span:.1f}s → {total_duration:.1f}s "
              f"(scale {scale:.3f}×) — SRT was from a stretched run")
        for s in segments:
            s.start *= scale
            s.end *= scale

    lang = language_code(args.language)
    voice = _resolve_voice(args.voice, lang, cfg)
    style = resolve_style("standard")
    voiceover = not args.no_voiceover
    mix_volume, orig_vol, gap_vol = _voiceover_volumes(voiceover, cfg)

    tmp_dir = Path(tempfile.mkdtemp(prefix="vidtrans_resume_"))

    # Resolve TTS dir: either reuse existing (skip TTS) or write fresh.
    if args.tts_dir:
        tts_dir = Path(args.tts_dir).expanduser().resolve()
        if not tts_dir.is_dir():
            print(f"[resume] --tts-dir {tts_dir} does not exist — abort.", file=sys.stderr)
            return 1
        keep_tts = True   # never delete a user-supplied cache dir
    elif args.keep_tts_dir:
        tts_dir = Path(args.keep_tts_dir).expanduser().resolve()
        tts_dir.mkdir(parents=True, exist_ok=True)
        keep_tts = True
    else:
        tts_dir = tmp_dir / "tts"
        tts_dir.mkdir()
        keep_tts = False

    tracker = CostTracker()
    tracker.start_timer()
    tracker.audio_minutes = total_duration / 60.0

    try:
        print(f"[resume] Preparing merge plan (voiceover={voiceover})…")
        plan = prepare_merge(
            args.input, segments, total_duration,
            preserve_gap_audio=voiceover,
            mix_volume=mix_volume,
            original_audio_volume=orig_vol,
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

        tts_label = cfg["models"]["tts"]["model"]
        if args.tts_dir:
            # Reuse cached TTS WAVs — build paths by segment id, verify each exists.
            tts_paths = [str(tts_dir / f"seg_{seg.id:05d}.wav") for seg in segments]
            missing = [p for p in tts_paths if not Path(p).is_file()]
            if missing:
                print(f"[resume] ERROR: {len(missing)} TTS files missing in {tts_dir} "
                      f"(first: {missing[0]}). Re-run without --tts-dir to regenerate.",
                      file=sys.stderr)
                return 2
            print(f"[resume] [TTS] reusing {len(tts_paths)} WAVs from {tts_dir} (skipping synthesis)")
            tracker.record_step(f"TTS (cached from {tts_dir.name})")
        else:
            print(f"[resume] [TTS] {tts_label} (voice: {voice}) → {tts_dir}"
                  f"{' (will keep)' if keep_tts else ''}…")
            tts_paths = synthesize_segments(
                segments, voice, str(tts_dir),
                language=lang,
                tracker=tracker,
                speed=style.tts_speed,
                emotion=style.tts_emotion,
            )
            tracker.record_step(f"TTS ({tts_label})")
        gap_thread.join()
        if gap_exc:
            raise gap_exc[0]

        print(f"[resume] [Merge] Building aligned video → {args.output}")
        aligned = build_aligned_video(
            args.input, segments, tts_paths, total_duration, args.output,
            merge_plan=plan,
        )
        tracker.record_step("Build aligned video")

        if not args.no_subtitles:
            srt_out = str(Path(args.output).with_suffix(".srt"))
            generate_srt(aligned, srt_out)
            print(f"[resume]   SRT → {srt_out}")

    finally:
        # Only delete the tmp dir; if user asked to keep TTS at a separate path,
        # that path is outside tmp_dir and not affected.
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n[resume] Done! → {args.output}")
    if keep_tts:
        print(f"[resume] TTS WAVs kept at: {tts_dir}")
        print(f"[resume]   → reuse with: --tts-dir {tts_dir}")
    tracker.print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
