"""Resume Violin pipeline from a persisted segments JSON.

The orchestrator writes ``<output>.transcribed.segments.json`` after Step 2,
``<output>.translated.segments.json`` after Step 3, and — on the fitter path
(``fit.enabled``) — ``<output>.fitted.segments.json`` after fit_text/fit_audio.
If the pipeline crashes during Translate / TTS / Merge, this script picks up
from the latest checkpoint and finishes the run — saving the (often 30-60
min) transcription work.

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

    # Resume from fitted segments (fit.enabled run; only TTS + merge, no re-split)
    uv run resume_from_segments.py \\
        --input  source.mp4 \\
        --segments output.fitted.segments.json \\
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
from dataclasses import asdict
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

    Returns (segments, stage) where stage is "transcribed", "translated", or
    "fitted" (translated + fitted: sentence units, ``end`` already extended
    to borrow the following pause — no re-merge/re-split needed).
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
            source_text=s.get("source_text", ""),
            words=s.get("words"),
        )
        for s in data["segments"]
    ]
    return segments, stage


def _voices_json_path(segments_path: str, stage: str) -> Path | None:
    """The `<stem>.voices.json` sibling of a `<stem>.<stage>.segments.json` file, if any.

    Mirrors how ``pipeline.orchestrator._persist_segments``/the diarization step
    name their artifacts — strips the exact `.{stage}.segments.json` suffix
    written for this checkpoint and reattaches `.voices.json`. Returns None
    when *segments_path* doesn't have that expected suffix.
    """
    p = Path(segments_path)
    suffix = f".{stage}.segments.json"
    if not p.name.endswith(suffix):
        return None
    stem = p.name[: -len(suffix)]
    return p.with_name(f"{stem}.voices.json")


def _load_voice_map(path: Path) -> dict[str, str] | None:
    """Load a persisted ``{speaker: voice}`` map, tolerating a missing/corrupt sidecar file.

    This script exists specifically to recover from a crashed run, so a
    truncated/empty ``voices.json``, literal ``null``, or any other non-dict
    JSON must not raise and kill the resume — it degrades to ``None`` (same
    as "no voices.json was ever written") with a one-line stderr warning.
    Entries whose key or value isn't a non-empty string are dropped the same
    way (keeping the rest of the map).
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"[resume] WARN: could not read voice_map {path}: {exc} — ignoring", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"[resume] WARN: voice_map {path} is not a JSON object ({type(data).__name__}) — ignoring",
              file=sys.stderr)
        return None
    voice_map = {
        k: v for k, v in data.items()
        if isinstance(k, str) and k and isinstance(v, str) and v
    }
    dropped = len(data) - len(voice_map)
    if dropped:
        noun = "entry" if dropped == 1 else "entries"
        print(f"[resume] WARN: dropped {dropped} invalid {noun} from voice_map {path}", file=sys.stderr)
    return voice_map


_RESUMABLE_STAGES = {"transcribed", "translated", "fitted"}


def _check_fit_stage(cfg: dict, stage: str) -> str | None:
    """Return an error message when resuming from *stage* would be unsafe.

    Checkpoints other than "transcribed"/"translated"/"fitted" — e.g.
    "diarized" or "sentences" — still hold source-language text; feeding
    them straight to TTS would dub the video in the wrong language. This
    check applies regardless of ``fit.enabled``.

    ``fit.enabled`` configs (local presets) additionally rely on the fitter's
    budgets, shortening, and pause borrowing to line up TTS with the source
    timing. Resuming from a "transcribed" or "translated" checkpoint would
    skip that stage entirely and go straight to plain TTS + merge, silently
    dropping the fit — only the "fitted" checkpoint already carries its
    output.
    """
    if stage not in _RESUMABLE_STAGES:
        return (
            f"'{stage}' is not a checkpoint this script can resume from "
            f"(expected one of {sorted(_RESUMABLE_STAGES)}). Earlier checkpoints "
            "like 'diarized' or 'sentences' still hold source-language text — "
            "feeding them to TTS would produce a garbage dub."
        )
    if cfg.get("fit", {}).get("enabled") and stage != "fitted":
        return (
            f"config has fit.enabled=true, but --segments is a '{stage}' checkpoint. "
            "Resuming from transcribed/translated segments would skip the duration "
            "fitter (budgets, shortening, pause borrowing) entirely. Either pass "
            "--config pointing at a config with fit.enabled: false, or resume from "
            "the *.fitted.segments.json artifact instead."
        )
    return None


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

    voice_map: dict[str, str] | None = None
    voices_json_path = _voices_json_path(args.segments, stage)
    if voices_json_path is not None:
        voice_map = _load_voice_map(voices_json_path)
        if voice_map is not None:
            print(f"[resume] voice_map loaded ← {voices_json_path} ({len(voice_map)} speakers)")

    fit_error = _check_fit_stage(cfg, stage)
    if fit_error:
        print(f"[resume] ERROR: {fit_error}", file=sys.stderr)
        return 2

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
                    "segments": [asdict(s) for s in translated],
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"      [persist] translated → {persist_path}")
            # Same post-translate transformation as orchestrator.
            translated = merge_continuous_segments(translated, max_duration=float("inf"))
            translated = split_into_sentences(translated)
        elif stage == "fitted":
            # Already translated AND fitted — units are already sentence-level
            # with `end` extended to borrow the following pause; skip the
            # re-merge/re-split (would re-flatten the borrowed pauses).
            print(f"[resume] [Translate] skipping (stage=fitted, {len(segments)} segments)")
            translated = segments
            tracker.record_step("Translation (cached)")
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
                voice_map=voice_map,
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
