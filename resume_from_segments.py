"""Resume Violin pipeline from a persisted segments JSON.

The orchestrator writes ``<output>.transcribed.segments.json`` after Step 2
and ``<output>.translated.segments.json`` after Step 3. If the pipeline
crashes during Translate / TTS / Merge, this script picks up from the
latest checkpoint and finishes the run — saving the (often 30-60 min)
transcription work.

Usage examples:

    # Resume from English transcribed segments (re-runs translate + TTS + merge)
    uv run resume_from_segments.py \\
        --input  source.mp4 \\
        --segments output.transcribed.segments.json \\
        --output  output_vi.mp4 \\
        --language Vietnamese \\
        --style    academic \\
        --config   config/local_mac.yaml

    # Resume from already-translated Vietnamese segments (only TTS + merge)
    uv run resume_from_segments.py \\
        --input  source.mp4 \\
        --segments output.translated.segments.json \\
        --output  output_vi.mp4 \\
        --language Vietnamese \\
        --style    academic \\
        --config   config/local_mac.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import threading
from pathlib import Path

from pipeline import config as pipeline_config
from pipeline.costs import CostTracker
from pipeline.extractor import get_video_duration
from pipeline.languages import language_code
from pipeline.llm_client import make_translation_client
from pipeline.merger import build_aligned_video, build_gap_chunks, generate_srt, prepare_merge
from pipeline.styles import resolve as resolve_style
from pipeline.transcriber import Segment, merge_continuous_segments, split_into_sentences
from pipeline.translator import translate_segments
from pipeline.tts import native_voices_for, synthesize_segments


def load_segments(path: str) -> tuple[list[Segment], str]:
    """Load segments from the JSON written by orchestrator._persist_segments.

    Returns (segments, stage) where stage is "transcribed" or "translated".
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    stage = data.get("stage", "transcribed")
    segments = [
        Segment(
            id=s["id"],
            start=float(s["start"]),
            end=float(s["end"]),
            text=s["text"],
            speaker=s.get("speaker", "SPEAKER_00"),
        )
        for s in data["segments"]
    ]
    return segments, stage


def _resolve_voice(voice: str | None, lang_code: str, cfg: dict) -> str:
    if voice:
        return voice
    male, female = native_voices_for(lang_code)
    gender = cfg.get("preferences", {}).get("voice_gender", "male").lower()
    return male if gender == "male" else female


def _voiceover_volumes(voiceover: bool, cfg: dict) -> tuple[float, float, float]:
    """Match orchestrator._voiceover_volumes (CLI bake mode)."""
    if not voiceover:
        return 0.0, 0.0, 1.0
    vo_volume = float(cfg["merge_video"].get("voiceover_volume", 0.1))
    gap_vol = min(1.0, 2 * vo_volume)
    return vo_volume, 0.0, gap_vol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Source video file")
    parser.add_argument("--segments", required=True, help="Path to *.segments.json from a prior run")
    parser.add_argument("--output", required=True, help="Output dubbed .mp4")
    parser.add_argument("--language", "-l", required=True, help="Target language name (e.g. Vietnamese)")
    parser.add_argument("--style", "-s", default="standard", help="Style profile (standard/academic/news/etc.)")
    parser.add_argument("--voice", "-v", default=None, help="TTS voice override (default: native)")
    parser.add_argument("--config", "-c", default="config/local_mac.yaml", help="Pipeline config YAML")
    parser.add_argument("--source-language", default="auto-detect")
    parser.add_argument("--no-voiceover", action="store_true")
    parser.add_argument("--no-subtitles", action="store_true")
    parser.add_argument("--tts-dir", default=None, help="Reuse TTS WAVs from this directory (skips TTS)")
    parser.add_argument("--keep-tts-dir", default=None, help="Persist TTS WAVs at this path for future merge-only re-runs")
    args = parser.parse_args()

    pipeline_config.load(args.config)
    cfg = pipeline_config.get()

    segments, stage = load_segments(args.segments)
    print(f"[resume] {stage} segments loaded: {len(segments)} from {args.segments}")
    if not segments:
        print("[resume] No segments — abort.", file=sys.stderr)
        return 1

    total_duration = get_video_duration(args.input)
    lang = language_code(args.language)
    style = resolve_style(args.style)
    voice = _resolve_voice(args.voice, lang, cfg)
    voiceover = not args.no_voiceover
    mix_volume, orig_vol, gap_vol = _voiceover_volumes(voiceover, cfg)

    tracker = CostTracker()
    tracker.start_timer()
    tracker.audio_minutes = total_duration / 60.0

    tmp_dir = Path(tempfile.mkdtemp(prefix="vidtrans_seg_"))

    # TTS dir routing (same logic as resume_from_srt.py)
    if args.tts_dir:
        tts_dir = Path(args.tts_dir).expanduser().resolve()
        if not tts_dir.is_dir():
            print(f"[resume] --tts-dir {tts_dir} not found — abort.", file=sys.stderr)
            return 1
        keep_tts = True
    elif args.keep_tts_dir:
        tts_dir = Path(args.keep_tts_dir).expanduser().resolve()
        tts_dir.mkdir(parents=True, exist_ok=True)
        keep_tts = True
    else:
        tts_dir = tmp_dir / "tts"
        tts_dir.mkdir()
        keep_tts = False

    try:
        # ── Step 3: Translate (only if stage is "transcribed") ─────────────
        if stage == "transcribed":
            print(f"[resume] [Translate] {len(segments)} segments → {args.language} (style: {style.name})")
            translation_client = make_translation_client(cfg)
            translated = translate_segments(
                segments, args.language, translation_client, args.source_language,
                tracker=tracker,
                style_directives=style.translation_directives,
                style_temperature=style.temperature,
            )
            tracker.record_step("Translation (LLM)")
            # Persist for next-stage crash recovery.
            persist_path = Path(args.output).with_suffix(".translated.segments.json")
            persist_path.write_text(
                json.dumps({
                    "stage": "translated",
                    "count": len(translated),
                    "segments": [
                        {"id": s.id, "start": s.start, "end": s.end,
                         "text": s.text, "speaker": s.speaker}
                        for s in translated
                    ],
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"      [persist] translated → {persist_path}")
            # Same post-translate transformation as orchestrator.
            translated = merge_continuous_segments(translated, max_duration=float("inf"))
            translated = split_into_sentences(translated)
        else:
            # Already translated — apply the same merge+split as orchestrator does
            # post-translate, since persisted JSON has pre-split text.
            print(f"[resume] [Translate] skipping (stage=translated, {len(segments)} segments)")
            translated = merge_continuous_segments(segments, max_duration=float("inf"))
            translated = split_into_sentences(translated)
            tracker.record_step("Translation (cached)")

        # ── Step 4: TTS ────────────────────────────────────────────────────
        print(f"[resume] [TTS] {cfg['models']['tts']['model']} (voice: {voice})")
        plan = prepare_merge(
            args.input, translated, total_duration,
            preserve_gap_audio=voiceover,
            mix_volume=mix_volume,
            original_audio_volume=orig_vol,
            gap_volume=gap_vol,
        )
        gap_exc: list[Exception] = []
        def _gap():
            try:
                build_gap_chunks(plan)
            except Exception as e:
                gap_exc.append(e)
        gap_thread = threading.Thread(target=_gap, daemon=True)
        gap_thread.start()

        if args.tts_dir:
            tts_paths = [str(tts_dir / f"seg_{s.id:05d}.wav") for s in translated]
            missing = [p for p in tts_paths if not Path(p).is_file()]
            if missing:
                print(f"[resume] ERROR: {len(missing)} TTS WAVs missing (first: {missing[0]})", file=sys.stderr)
                return 2
            print(f"[resume]   reusing {len(tts_paths)} TTS WAVs from {tts_dir}")
            tracker.record_step("TTS (cached)")
        else:
            tts_paths = synthesize_segments(
                translated, voice, str(tts_dir),
                language=lang,
                tracker=tracker,
                speed=style.tts_speed,
                emotion=style.tts_emotion,
            )
            tracker.record_step(f"TTS ({cfg['models']['tts']['model']})")
        gap_thread.join()
        if gap_exc:
            raise gap_exc[0]

        # ── Step 5: Merge ──────────────────────────────────────────────────
        print(f"[resume] [Merge] → {args.output}")
        aligned = build_aligned_video(
            args.input, translated, tts_paths, total_duration, args.output,
            merge_plan=plan,
        )
        tracker.record_step("Build aligned video")

        if not args.no_subtitles:
            srt_out = str(Path(args.output).with_suffix(".srt"))
            generate_srt(aligned, srt_out)
            print(f"[resume]   SRT → {srt_out}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n[resume] Done! → {args.output}")
    if keep_tts:
        print(f"[resume] TTS WAVs kept at: {tts_dir}")
    tracker.print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
