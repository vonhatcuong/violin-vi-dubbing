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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

from . import config as pipeline_config
from . import diarizer
from . import fitter
from .costs import CostTracker
from .extractor import ensure_video_input, extract_audio, get_video_duration
from .languages import language_code
from .llm_client import make_transcription_client, make_translation_client
from .merger import burn_subtitles, build_aligned_video, build_gap_chunks, generate_subtitle_files, generate_transcript, prepare_merge
from .styles import StyleProfile, resolve as resolve_style
from .subtitles import split_into_cues
from .timemap import build_time_map
from .transcriber import Segment, merge_continuous_segments, split_into_sentences, split_long_segments, transcribe
from .translator import shorten_segment, translate_segments
from .tts import make_batch_synthesizer, make_synthesizer, native_voices_for, synthesize_segments
from .voices import assign_voices, guess_genders

ProgressCallback = Callable[[int, str], None]
CancelCallback = Callable[[], bool]


class Cancelled(Exception):
    """Raised inside ``dub_video`` when ``is_cancelled()`` returns True."""


def valid_speakers_value(value: str) -> bool:
    """True for "auto", or a positive integer string with no leading zero (e.g. "3", not "03").

    Shared by ``DubOptions.__post_init__`` and ``main.py``'s ``--speakers`` argparse
    validator so the two can't drift.
    """
    if value == "auto":
        return True
    return value.isascii() and value.isdigit() and value == str(int(value)) and int(value) >= 1


@dataclass
class DubOptions:
    target_language: str
    source_language: str = "auto-detect"
    voice: str | None = None              # None → pick native by preferences.voice_gender
    style: StyleProfile | None = None     # None → resolve "standard"
    voiceover: bool = True                # mix original audio with the dub
    bake_voiceover: bool = True           # True (CLI): bake into video; False (API): export separate track
    subtitles: bool = True                # generate SRT alongside the video
    subtitle_lang: str | None = None      # "target" (translated) | "source" (original ASR sentences re-timed); None → config subtitles.language
    subtitle_formats: tuple[str, ...] = ("srt",)
    burn_subtitles: bool = False          # create a second video with subtitles burned in
    prefer_source_captions: bool = True   # URL inputs: prefer YouTube captions over Whisper
    fit: bool | None = None               # None → config fit.enabled; duration fitter (local dubbing)
    speakers: str = "1"                   # "1" (off) | "auto" (auto-detect count) | "N" (fixed speaker count)
    voice_map: dict[str, str] | None = None  # explicit SPEAKER_xx → voice name overrides

    # BYOK overrides (used by the web app when a user supplies their own keys)
    together_api_key: str | None = None
    openai_api_key: str | None = None
    elevenlabs_api_key: str | None = None

    def __post_init__(self) -> None:
        if not valid_speakers_value(self.speakers):
            raise ValueError(
                f'DubOptions.speakers must be "auto" or a positive integer string (no leading zero), '
                f'got {self.speakers!r}'
            )


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
            audio_path: str | None = None
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
        fit_cfg = cfg.get("fit", {})
        fit_enabled = bool(fit_cfg.get("enabled", False)) if opts.fit is None else bool(opts.fit)
        effective_voice = _resolve_voice(opts.voice, lang_code, cfg)

        # ── Diarization + per-speaker voice assignment (Task 24) ──
        # Must run before merge_continuous_segments: merging only glues same-speaker
        # segments, so mislabeling here would let it join two different speakers.
        vcfg = cfg.get("voices", {})
        dcfg = cfg.get("diarization", {})
        tts_provider = cfg.get("models", {}).get("tts", {}).get("provider")
        voice_map: dict[str, str] = {}
        if opts.speakers != "1" or bool(dcfg.get("enabled", False)):
            _check_cancel(is_cancelled)
            if audio_path is None:
                audio_path = extract_audio(input_path, str(tmp_dir / "audio.wav"))
            _emit(on_progress, 2, "Diarizing speakers…")
            hf_token_env = dcfg.get("hf_token_env")
            if opts.speakers == "auto":
                num_speakers = None
            elif opts.speakers == "1":
                # Only reachable via diarization.enabled with no explicit --speakers —
                # never force a single cluster; fall back to config or full auto-detect.
                num_speakers = dcfg.get("num_speakers") or None
            else:
                num_speakers = int(opts.speakers)
            labels = diarizer.label_segments(
                audio_path, segments,
                backend=dcfg.get("backend", "ecapa"),
                num_speakers=num_speakers,
                max_speakers=int(dcfg.get("max_speakers", 4)),
                threshold=float(dcfg.get("threshold", 0.65)),
                min_cluster_segments=int(dcfg.get("min_cluster_segments", 3)),
                min_cluster_seconds=float(dcfg.get("min_cluster_seconds", 3.0)),
                hf_token=os.environ.get(hf_token_env) if hf_token_env else None,
                model=dcfg.get("model", "speechbrain/spkrec-ecapa-voxceleb"),
                pyannote_model=dcfg.get("pyannote_model", "pyannote/speaker-diarization-community-1"),
                device=dcfg.get("device", "auto"),
            )
            segments = [replace(s, speaker=lab) for s, lab in zip(segments, labels)]
            _persist_segments(segments, output_video_path, "diarized")
            tracker.record_step("Diarization")

            # speaker_voices/preset_genders/genders only make sense for VieNeu preset
            # names — other providers (e.g. Cartesia/Together) would get sent a VieNeu
            # preset name as a literal voice id, so every speaker just gets
            # effective_voice there unless --voice-map names them explicitly.
            speakers_order = list(dict.fromkeys(s.speaker for s in segments))
            if tts_provider == "vieneu":
                genders: dict[str, str] = {}
                if vcfg.get("gender_detect", False):
                    genders = guess_genders(audio_path, segments)
                voice_map = assign_voices(
                    speakers_order, effective_voice, opts.voice_map, genders,
                    speaker_voices=vcfg.get("speaker_voices"), preset_genders=vcfg.get("preset_genders"),
                    seed_voice=opts.voice,
                )
            else:
                voice_map = assign_voices(
                    speakers_order, effective_voice, opts.voice_map, seed_voice=opts.voice,
                )
            voices_json_path = Path(output_video_path).with_suffix(".voices.json")
            voices_json_path.write_text(json.dumps(voice_map, ensure_ascii=False, indent=2), encoding="utf-8")

        raw_sentences = [replace(s) for s in segments]
        _persist_segments(raw_sentences, output_video_path, "sentences")

        segments = merge_continuous_segments(segments)
        tcfg = cfg.get("transcription", {})
        segments = split_long_segments(
            segments, float(tcfg.get("max_sentence_seconds", 0) or 0), float(tcfg.get("min_piece_seconds", 2.5)),
            long_gap_seconds=float(tcfg.get("long_gap_seconds", 2.0)),
        )
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

        if fit_enabled and fit_cfg.get("pipelined"):
            # Overlap translation batch N+1 (LLM) with shortening + TTS of batch N
            # (VieNeu) — different devices, so both run concurrently.
            tts_label = cfg["models"]["tts"]["model"]
            _emit(on_progress, 3,
                  f"Translating + synthesizing {len(segments)} segments to {opts.target_language} "
                  f"(pipelined, style: {style.name})…")
            tts_dir = tmp_dir / "tts"
            tts_dir.mkdir()

            mix_volume, original_audio_volume, gap_vol = _voiceover_volumes(opts, cfg)

            def _shorten(src: str, cur: str, budget_syll: int, budget_s: float) -> str:
                return shorten_segment(src, cur, budget_syll, budget_s, opts.target_language,
                                       translation_client, tracker=tracker,
                                       source_language=opts.source_language)

            def _translate_batch(batch: list[Segment], batch_budgets: list[tuple[float, int]]) -> list[Segment]:
                return translate_segments(
                    batch, opts.target_language, translation_client, opts.source_language,
                    tracker=tracker,
                    style_directives=style.translation_directives,
                    style_temperature=style.temperature,
                    budgets=batch_budgets,
                )

            def _on_pipelined_batch(partial_translated: list[Segment], partial_units: list, total: int) -> None:
                # Mirrors the sequential branch: check cancellation and persist the
                # translated checkpoint incrementally instead of only at the end.
                _check_cancel(is_cancelled)
                _persist_segments(partial_translated, output_video_path, "translated")

            synth = make_synthesizer(language=lang_code, emotion=style.tts_emotion)
            synth_batch = make_batch_synthesizer(language=lang_code, emotion=style.tts_emotion)
            translated, units = fitter.run_pipelined(
                segments, slots, _translate_batch, _shorten, synth, synth_batch, str(tts_dir),
                dict(fit_cfg, _voice_map=voice_map, _default_voice=effective_voice),
                batch_size=int(cfg["translation"].get("batch_size", 32)),
                workers=int(cfg["translation"].get("parallel_batches", 1)),
                on_batch=_on_pipelined_batch,
            )
            tracker.record_step(f"Translation + TTS + fit (pipelined, {tts_label})")
            _persist_segments(translated, output_video_path, "translated")

            translated, tts_paths = fitter.apply_units(units, translated)
            fitter.save_units(units, Path(output_video_path).with_suffix(".fit.units.json"))
            _persist_segments(translated, output_video_path, "fitted")
            plan = prepare_merge(
                video_input_path, translated, total_duration,
                preserve_gap_audio=opts.voiceover,
                mix_volume=mix_volume,
                original_audio_volume=original_audio_volume,
                gap_volume=gap_vol,
            )
            build_gap_chunks(plan)
        else:
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
            tts_label = cfg["models"]["tts"]["model"]
            _emit(on_progress, 4, f"Synthesizing TTS with {tts_label} (voice: {effective_voice})…")
            tts_dir = tmp_dir / "tts"
            tts_dir.mkdir()

            mix_volume, original_audio_volume, gap_vol = _voiceover_volumes(opts, cfg)

            if fit_enabled:
                units = fitter.build_units(translated, slots, voice_map, effective_voice)

                def _shorten(src: str, cur: str, budget_syll: int, budget_s: float) -> str:
                    return shorten_segment(src, cur, budget_syll, budget_s, opts.target_language,
                                           translation_client, tracker=tracker,
                                           source_language=opts.source_language)

                fitter.fit_text(units, _shorten, fit_cfg)
                synth = make_synthesizer(language=lang_code, emotion=style.tts_emotion)
                synth_batch = make_batch_synthesizer(language=lang_code, emotion=style.tts_emotion)
                fitter.fit_audio(units, synth, str(tts_dir), fit_cfg, synth_batch=synth_batch)
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
                    voice_map=voice_map,
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

        sub_lang = (opts.subtitle_lang or cfg.get("subtitles", {}).get("language", "target")).lower()
        if sub_lang == "source":
            scfg = cfg.get("subtitles", {})
            cues = split_into_cues(
                raw_sentences,
                max_chars=scfg.get("max_chars", 84),
                max_duration=float(scfg.get("max_duration", 6.0)),
                min_duration=float(scfg.get("min_duration", 1.0)),
            )
            tmap = build_time_map(translated, aligned_segments)
            subtitle_segments = [
                Segment(id=i, start=tmap(s.start), end=tmap(s.end), text=s.text, speaker=s.speaker)
                for i, s in enumerate(cues)
            ]
        else:
            subtitle_segments = aligned_segments
        subtitle_paths: dict[str, str] = {}
        if output_srt_path is not None and opts.subtitles:
            subtitle_paths = generate_subtitle_files(subtitle_segments, output_srt_path, formats=opts.subtitle_formats)
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
