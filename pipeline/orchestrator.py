"""Shared dubbing pipeline used by both the CLI and the FastAPI worker.

Both entry points construct ``DubOptions`` and call ``dub_video``. The two
callbacks (``on_progress`` / ``is_cancelled``) carry the only meaningful
asynchronous differences between the two surfaces — everything else is the
same five-step flow.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import config as pipeline_config
from .costs import CostTracker
from .extractor import ensure_video_input, extract_audio, get_video_duration
from .languages import language_code
from .llm_client import make_transcription_client, make_translation_client
from .merger import burn_subtitles, build_aligned_video, build_gap_chunks, generate_subtitle_files, generate_transcript, prepare_merge
from .styles import StyleProfile, resolve as resolve_style
from .transcriber import Segment, merge_continuous_segments, split_into_sentences, transcribe
from .translator import translate_segments
from .tts import native_voices_for, synthesize_segments

ProgressCallback = Callable[[int, str], None]
CancelCallback = Callable[[], bool]


class Cancelled(Exception):
    """Raised inside ``dub_video`` when ``is_cancelled()`` returns True."""


@dataclass
class DubOptions:
    target_language: str
    source_language: str = "auto-detect"
    voice: str | None = None              # None → pick native by preferences.voice_gender
    style: StyleProfile | None = None     # None → resolve "standard"
    voiceover: bool = True                # mix original audio with the dub
    bake_voiceover: bool = True           # True (CLI): bake into video; False (API): export separate track
    subtitles: bool = True                # generate SRT alongside the video
    subtitle_formats: tuple[str, ...] = ("srt",)
    burn_subtitles: bool = False          # create a second video with subtitles burned in
    prefer_source_captions: bool = True   # URL inputs: prefer YouTube captions over Whisper

    # BYOK overrides (used by the web app when a user supplies their own keys)
    together_api_key: str | None = None
    openai_api_key: str | None = None
    elevenlabs_api_key: str | None = None


@dataclass
class DubResult:
    aligned_segments: list[Segment]
    output_video_path: str
    output_srt_path: str | None
    cost_tracker: CostTracker
    subtitle_paths: dict[str, str] = field(default_factory=dict)
    transcript_path: str | None = None
    burned_video_path: str | None = None
    original_audio_path: str | None = None
    steps: list[dict] = field(default_factory=list)


def dub_video(
    input_path: str,
    output_video_path: str,
    opts: DubOptions,
    *,
    output_srt_path: str | None = None,
    transcript_path: str | None = None,
    burned_video_path: str | None = None,
    original_audio_path: str | None = None,
    on_progress: ProgressCallback | None = None,
    is_cancelled: CancelCallback | None = None,
    tracker: CostTracker | None = None,
    segments_override: list[Segment] | None = None,
) -> DubResult:
    """Run the full dubbing pipeline. Both the CLI and the API worker call this."""
    cfg = pipeline_config.get()
    style = opts.style if opts.style is not None else resolve_style("standard")
    tracker = tracker or CostTracker()

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

        lang_code = language_code(opts.target_language)
        segments = merge_continuous_segments(segments)
        _persist_segments(segments, output_video_path, "transcribed")

        _check_cancel(is_cancelled)
        _emit(on_progress, 3, f"Translating {len(segments)} segments to {opts.target_language} (style: {style.name})…")
        translated = translate_segments(
            segments, opts.target_language, translation_client, opts.source_language,
            tracker=tracker,
            style_directives=style.translation_directives,
            style_temperature=style.temperature,
        )
        tracker.record_step("Translation (LLM)")
        _persist_segments(translated, output_video_path, "translated")
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

        subtitle_paths: dict[str, str] = {}
        if output_srt_path is not None and opts.subtitles:
            subtitle_paths = generate_subtitle_files(
                aligned_segments,
                output_srt_path,
                formats=opts.subtitle_formats,
            )
        else:
            output_srt_path = None

        if transcript_path is None:
            transcript_path = str(Path(output_video_path).with_suffix(".transcript.txt"))
        generate_transcript(aligned_segments, transcript_path)

        if opts.burn_subtitles:
            srt_path = subtitle_paths.get("srt")
            if not srt_path:
                raise RuntimeError("Burned subtitles require SRT subtitle generation.")
            if burned_video_path is None:
                burned_video_path = str(Path(output_video_path).with_stem(Path(output_video_path).stem + "_subtitled"))
            burn_subtitles(output_video_path, srt_path, burned_video_path)
        else:
            burned_video_path = None

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return DubResult(
        aligned_segments=aligned_segments,
        output_video_path=output_video_path,
        output_srt_path=output_srt_path,
        subtitle_paths=subtitle_paths,
        transcript_path=transcript_path,
        burned_video_path=burned_video_path,
        original_audio_path=original_audio_path,
        cost_tracker=tracker,
        steps=list(tracker._steps),
    )


def _persist_segments(segments: list[Segment], output_video_path: str, stage: str) -> None:
    """Dump segments to JSON next to the output video for crash recovery.

    Writes ``<output>.{stage}.segments.json`` (atomic via temp-rename). Failures
    are non-fatal — persistence is a recovery convenience, not a hard
    requirement of the pipeline.
    """
    try:
        out = Path(output_video_path).with_suffix(f".{stage}.segments.json")
        payload = {"stage": stage, "count": len(segments), "segments": [asdict(s) for s in segments]}
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, out)
        print(f"      [persist] {stage} → {out} ({len(segments)} segments)")
    except Exception as exc:
        print(f"      [persist] WARN: failed to dump {stage} segments: {exc}")


def _check_cancel(cb: CancelCallback | None) -> None:
    if cb is not None and cb():
        raise Cancelled()


def _emit(cb: ProgressCallback | None, step: int, msg: str) -> None:
    if cb is not None:
        cb(step, msg)


def _resolve_voice(voice: str | None, lang_code: str, cfg: dict) -> str:
    if voice:
        return voice
    gender_idx = 0 if cfg["preferences"].get("voice_gender", "male") == "male" else 1
    return native_voices_for(lang_code)[gender_idx]


def _voiceover_volumes(opts: DubOptions, cfg: dict) -> tuple[float, float, float]:
    """Translate (voiceover, bake_voiceover) into the three volume knobs prepare_merge wants."""
    if not opts.voiceover:
        return 0.0, 0.0, 1.0
    vo_volume = cfg["merge_video"].get("voiceover_volume", 0.35)
    gap_vol = min(1.0, 2 * vo_volume)
    if opts.bake_voiceover:
        # CLI mode: original audio baked into final video at vo_volume.
        return vo_volume, 0.0, gap_vol
    # API mode: video has pure dub; caller exports original audio as a separate
    # track that the browser overlays at user-controlled volume.
    return 0.0, 1.0, gap_vol
