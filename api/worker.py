"""Background worker that runs the dubbing pipeline in a thread pool."""

from __future__ import annotations

import time as _time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from pipeline import config as pipeline_config
from pipeline.captions import fetch_source_captions
from pipeline.downloader import download_url_to_file
from pipeline.llm_client import get_transcription_provider, get_translation_provider
from pipeline.orchestrator import Cancelled, DubOptions, dub_video
from pipeline.styles import resolve as resolve_style
from pipeline.tts import get_tts_provider

from . import stats as _stats
from .config import MAX_WORKERS
from .models import JobStatus
from .storage import (
    _read_meta,
    append_progress,
    burned_video_path,
    input_path,
    original_audio_path,
    output_srt_path,
    output_video_path,
    save_segments,
    transcript_path,
    update_status,
)

load_dotenv(override=True)

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

TOTAL_STEPS = 5


def _is_cancelled(job_id: str) -> bool:
    try:
        meta = _read_meta(job_id)
        return meta.get("status") in (JobStatus.cancelled, "cancelled")
    except (OSError, ValueError):
        return False


def _run_job(
    job_id: str,
    params: dict,
    together_key_override: str | None = None,
    openai_key_override: str | None = None,
    elevenlabs_key_override: str | None = None,
    segments_override=None,
) -> None:
    update_status(job_id, JobStatus.running)

    target_language = params["language"]
    voice = params["voice"]
    source_language = params["source_language"]
    subtitles = params["subtitles"]
    subtitle_formats = tuple(params.get("subtitle_formats") or ("srt",))
    burn_subs = bool(params.get("burn_subtitles", False))
    voiceover = params.get("voiceover", True)
    style = resolve_style(params.get("style", "standard"))

    started_at = int(_time.time())
    segments_count = 0
    total_duration = 0.0
    error_msg: str | None = None
    final_status = JobStatus.done
    tracker = None

    src = input_path(job_id)
    out_video = output_video_path(job_id)
    out_srt = output_srt_path(job_id) if subtitles else None
    orig_audio = str(original_audio_path(job_id)) if voiceover else None

    opts = DubOptions(
        target_language=target_language,
        source_language=source_language,
        voice=voice or None,
        style=style,
        voiceover=voiceover,
        bake_voiceover=False,           # web mode: browser overlays original at user-chosen volume
        subtitles=subtitles,
        subtitle_formats=subtitle_formats,
        burn_subtitles=burn_subs,
        together_api_key=together_key_override,
        openai_api_key=openai_key_override,
        elevenlabs_api_key=elevenlabs_key_override,
    )

    try:
        result = dub_video(
            str(src),
            str(out_video),
            opts,
            output_srt_path=str(out_srt) if out_srt else None,
            transcript_path=str(transcript_path(job_id)),
            burned_video_path=str(burned_video_path(job_id)) if burn_subs else None,
            original_audio_path=orig_audio,
            on_progress=lambda step, msg: append_progress(job_id, step, TOTAL_STEPS, msg),
            is_cancelled=lambda: _is_cancelled(job_id),
            segments_override=segments_override,
        )
        tracker = result.cost_tracker
        total_duration = (tracker.audio_minutes or 0.0) * 60.0
        segments_count = len(result.aligned_segments)
        save_segments(
            job_id,
            [
                {
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker": seg.speaker,
                }
                for seg in result.aligned_segments
            ],
        )
        update_status(job_id, JobStatus.done)

    except Cancelled:
        final_status = JobStatus.cancelled
    except Exception as exc:
        error_msg = str(exc)
        final_status = JobStatus.failed
        update_status(job_id, JobStatus.failed, error_msg)

    # Persist stats — never crashes the pipeline. Skip cancelled jobs.
    if final_status != JobStatus.cancelled:
        try:
            cfg = pipeline_config.get()
            cb = tracker.cost_breakdown() if tracker else {
                "whisper": {"cost": 0}, "translation": {"cost": 0},
                "tts": {"cost": 0}, "total": 0,
            }
            finished_at = int(_time.time())
            _stats.record_job({
                "id": job_id,
                "created_at": started_at,
                "finished_at": finished_at,
                "status": final_status.value,
                "target_language": target_language,
                "source_language": source_language,
                "style": style.name,
                "voice": voice or "",
                "voiceover": 1 if voiceover else 0,
                "transcription_provider": get_transcription_provider(cfg),
                "translation_provider": get_translation_provider(cfg),
                "tts_provider": get_tts_provider(),
                "segments_count": segments_count,
                "audio_seconds": total_duration,
                "tts_characters": tracker.tts_characters if tracker else 0,
                "llm_input_tokens": tracker.llm_input_tokens if tracker else 0,
                "llm_output_tokens": tracker.llm_output_tokens if tracker else 0,
                "whisper_cost_usd": cb["whisper"]["cost"],
                "translation_cost_usd": cb["translation"]["cost"],
                "tts_cost_usd": cb["tts"]["cost"],
                "total_cost_usd": cb["total"],
                "duration_seconds": finished_at - started_at,
                "error": error_msg,
            })
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Failed to record stats")


def _download_url(job_id: str, url: str) -> Path:
    """Download a video from a URL using yt-dlp. Returns the path to the downloaded file."""
    from .config import MAX_DURATION_SECONDS, MAX_FILE_SIZE_MB
    from .storage import _job_dir

    return download_url_to_file(
        url,
        _job_dir(job_id),
        max_duration_seconds=MAX_DURATION_SECONDS,
        max_file_size_mb=MAX_FILE_SIZE_MB,
    )


def _run_url_job(
    job_id: str,
    params: dict,
    url: str,
    together_key_override: str | None = None,
    openai_key_override: str | None = None,
    elevenlabs_key_override: str | None = None,
) -> None:
    """Download video from URL, then run the normal translation pipeline."""
    update_status(job_id, JobStatus.running)
    append_progress(job_id, 1, TOTAL_STEPS, "Downloading video from URL…")

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


def submit_job(
    job_id: str,
    params: dict,
    *,
    together_key_override: str | None = None,
    openai_key_override: str | None = None,
    elevenlabs_key_override: str | None = None,
) -> None:
    """Submit a job to the thread pool for background execution."""
    _executor.submit(_run_job, job_id, params, together_key_override, openai_key_override, elevenlabs_key_override)


def submit_url_job(
    job_id: str,
    params: dict,
    url: str,
    *,
    together_key_override: str | None = None,
    openai_key_override: str | None = None,
    elevenlabs_key_override: str | None = None,
) -> None:
    """Submit a URL-based job to the thread pool."""
    _executor.submit(_run_url_job, job_id, params, url, together_key_override, openai_key_override, elevenlabs_key_override)
