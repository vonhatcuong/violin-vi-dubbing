"""File-based job storage.

Each job lives in JOBS_DIR/{job_id}/:
    meta.json       — JobStatus + parameters
    progress.jsonl  — append-only progress events (one JSON object per line)
    input.<ext>     — uploaded source video
    output.mp4      — dubbed video (present when status=done)
    output.srt      — subtitle file (present when status=done and subtitles=True)
    segments.json   — aligned subtitle/timeline segments for playback + chat context
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import JOBS_DIR
from .models import JobHistoryItem, JobResponse, JobStatus, ProgressEvent

logger = logging.getLogger(__name__)

_lock = threading.Lock()


# Patterns for stripping API keys out of error messages before persisting them
# to disk. Matches the known prefixes (OpenAI / ElevenLabs) and any sufficiently
# long hex / alphanumeric run that looks like a Together key.
_KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),       # OpenAI / Anthropic-style
    re.compile(r"sk_[A-Za-z0-9_-]{20,}"),       # ElevenLabs
    re.compile(r"\b[A-Fa-f0-9]{48,}\b"),        # Together (long hex token)
]


def _redact(text: str) -> str:
    """Strip anything that looks like an API key from a string."""
    for pat in _KEY_PATTERNS:
        text = pat.sub("***", text)
    return text


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _meta_path(job_id: str) -> Path:
    return _job_dir(job_id) / "meta.json"


def _progress_path(job_id: str) -> Path:
    return _job_dir(job_id) / "progress.jsonl"


def _history_db_path() -> Path:
    return JOBS_DIR / "jobs.sqlite"


_HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS job_history (
    id                 TEXT PRIMARY KEY,
    status             TEXT NOT NULL,
    language           TEXT NOT NULL,
    voice              TEXT NOT NULL DEFAULT '',
    source_language    TEXT NOT NULL DEFAULT 'auto-detect',
    subtitles          INTEGER NOT NULL DEFAULT 1,
    subtitle_formats   TEXT NOT NULL DEFAULT '[]',
    burn_subtitles     INTEGER NOT NULL DEFAULT 0,
    style              TEXT NOT NULL DEFAULT 'standard',
    voiceover          INTEGER NOT NULL DEFAULT 1,
    source_url         TEXT NOT NULL DEFAULT '',
    created_at         INTEGER NOT NULL,
    updated_at         INTEGER NOT NULL,
    error              TEXT,
    progress_count     INTEGER NOT NULL DEFAULT 0,
    deleted            INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_job_history_created_at ON job_history(created_at);
CREATE INDEX IF NOT EXISTS idx_job_history_status ON job_history(status);
"""


@contextmanager
def _history_conn() -> Iterator[sqlite3.Connection]:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_history_db_path()))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_HISTORY_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_job(job_id: str, params: dict[str, Any]) -> None:
    """Initialize a new job directory and meta.json."""
    job_dir = _job_dir(job_id)
    with _lock:
        job_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "id": job_id,
        "status": JobStatus.queued,
        "created_at": int(time.time()),
        **params,
        "error": None,
    }
    _atomic_write(_meta_path(job_id), json.dumps(meta))
    _progress_path(job_id).write_text("", encoding="utf-8")
    _upsert_history(meta)


def _queue_position(job_id: str, meta: dict) -> int:
    """Count other jobs (queued + running) created before this one.

    Returns 0 unless the target job itself is queued — running/done/failed/
    cancelled jobs aren't waiting in line anymore. Legacy jobs missing
    ``created_at`` also return 0 (can't compute reliably).
    """
    if meta.get("status") != JobStatus.queued:
        return 0
    my_created = meta.get("created_at", 0)
    if not my_created:
        return 0

    ahead = 0
    try:
        for entry in JOBS_DIR.iterdir():
            if not entry.is_dir() or entry.name == job_id:
                continue
            other_meta_path = entry / "meta.json"
            if not other_meta_path.exists():
                continue
            try:
                other = json.loads(other_meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if other.get("status") not in ("queued", "running"):
                continue
            other_created = other.get("created_at", 0)
            if other_created and other_created < my_created:
                ahead += 1
    except OSError:
        pass
    return ahead


def update_status(job_id: str, status: JobStatus, error: str | None = None) -> None:
    """Update the job status (and optionally record an error message)."""
    meta = _read_meta(job_id)
    meta["status"] = status
    if error is not None:
        meta["error"] = _redact(error)
    _atomic_write(_meta_path(job_id), json.dumps(meta))
    _upsert_history(meta)


def append_progress(job_id: str, step: int, total: int, message: str) -> None:
    """Append a progress event to the job's progress log."""
    event = json.dumps({"step": step, "total": total, "message": _redact(message)})
    with open(_progress_path(job_id), "a", encoding="utf-8") as f:
        f.write(event + "\n")
    _increment_history_progress(job_id)


def get_job(job_id: str) -> JobResponse | None:
    """Read job metadata and progress, returning None if the job doesn't exist."""
    meta_path = _meta_path(job_id)
    if not meta_path.exists():
        return None

    try:
        meta = _read_meta(job_id)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read meta for job %s: %s", job_id, exc)
        return None

    progress: list[ProgressEvent] = []
    progress_path = _progress_path(job_id)
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                progress.append(ProgressEvent(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue

    return JobResponse(
        id=meta["id"],
        status=meta["status"],
        language=meta["language"],
        voice=meta["voice"],
        source_language=meta["source_language"],
        subtitles=meta["subtitles"],
        subtitle_formats=_subtitle_formats_from_meta(meta),
        burn_subtitles=bool(meta.get("burn_subtitles", False)),
        style=meta.get("style", "standard"),
        voiceover=bool(meta.get("voiceover", True)),
        source_url=meta.get("source_url", ""),
        progress=progress,
        error=meta.get("error"),
        queue_position=_queue_position(job_id, meta),
    )


def input_path(job_id: str) -> Path:
    """Return the path where the uploaded video is stored."""
    job_dir = _job_dir(job_id)
    for p in job_dir.glob("input.*"):
        return p
    raise FileNotFoundError(f"No input file for job {job_id}")


def output_video_path(job_id: str) -> Path:
    return _job_dir(job_id) / "output.mp4"


def voiceover_video_path(job_id: str) -> Path:
    return _job_dir(job_id) / "output_voiceover.mp4"


def output_srt_path(job_id: str) -> Path:
    return _job_dir(job_id) / "output.srt"


def output_vtt_path(job_id: str) -> Path:
    return _job_dir(job_id) / "output.vtt"


def output_txt_path(job_id: str) -> Path:
    return _job_dir(job_id) / "output.txt"


def transcript_path(job_id: str) -> Path:
    return _job_dir(job_id) / "transcript.txt"


def burned_video_path(job_id: str) -> Path:
    return _job_dir(job_id) / "output_subtitled.mp4"


def original_audio_path(job_id: str) -> Path:
    return _job_dir(job_id) / "original_audio.m4a"


def segments_path(job_id: str) -> Path:
    return _job_dir(job_id) / "segments.json"


def save_segments(job_id: str, segments: list[dict[str, Any]]) -> None:
    segments_path(job_id).write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_segments(job_id: str) -> list[dict[str, Any]]:
    path = segments_path(job_id)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def delete_job(job_id: str) -> bool:
    """Remove a job directory. Returns True if the job existed."""
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        return False
    _mark_history_deleted(job_id)
    shutil.rmtree(job_dir)
    return True


def list_job_history(limit: int = 100) -> list[JobHistoryItem]:
    """Return recent job lifecycle history from SQLite."""
    with _lock, _history_conn() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM job_history
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_history_item_from_row(row) for row in rows]


def _upsert_history(meta: dict[str, Any]) -> None:
    now = int(time.time())
    created_at = int(meta.get("created_at") or now)
    subtitle_formats = meta.get("subtitle_formats") or ("srt",)
    if isinstance(subtitle_formats, str):
        subtitle_formats = [fmt.strip() for fmt in subtitle_formats.split(",") if fmt.strip()]
    row = {
        "id": meta["id"],
        "status": _status_value(meta.get("status", JobStatus.queued)),
        "language": meta.get("language", ""),
        "voice": meta.get("voice", ""),
        "source_language": meta.get("source_language", "auto-detect"),
        "subtitles": 1 if meta.get("subtitles", True) else 0,
        "subtitle_formats": json.dumps(list(subtitle_formats)),
        "burn_subtitles": 1 if meta.get("burn_subtitles", False) else 0,
        "style": meta.get("style", "standard"),
        "voiceover": 1 if meta.get("voiceover", True) else 0,
        "source_url": meta.get("source_url", ""),
        "created_at": created_at,
        "updated_at": now,
        "error": _redact(meta.get("error") or "") or None,
        "deleted": 0,
    }
    with _lock, _history_conn() as conn:
        conn.execute(
            """
            INSERT INTO job_history (
                id, status, language, voice, source_language, subtitles,
                subtitle_formats, burn_subtitles, style, voiceover, source_url,
                created_at, updated_at, error, progress_count, deleted
            )
            VALUES (
                :id, :status, :language, :voice, :source_language, :subtitles,
                :subtitle_formats, :burn_subtitles, :style, :voiceover, :source_url,
                :created_at, :updated_at, :error, 0, :deleted
            )
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status,
                language=excluded.language,
                voice=excluded.voice,
                source_language=excluded.source_language,
                subtitles=excluded.subtitles,
                subtitle_formats=excluded.subtitle_formats,
                burn_subtitles=excluded.burn_subtitles,
                style=excluded.style,
                voiceover=excluded.voiceover,
                source_url=excluded.source_url,
                updated_at=excluded.updated_at,
                error=excluded.error,
                deleted=0
            """,
            row,
        )


def _increment_history_progress(job_id: str) -> None:
    with _lock, _history_conn() as conn:
        conn.execute(
            """
            UPDATE job_history
            SET progress_count = progress_count + 1,
                updated_at = ?
            WHERE id = ?
            """,
            (int(time.time()), job_id),
        )


def _mark_history_deleted(job_id: str) -> None:
    with _lock, _history_conn() as conn:
        conn.execute(
            """
            UPDATE job_history
            SET deleted = 1,
                status = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (JobStatus.cancelled.value, int(time.time()), job_id),
        )


def _history_item_from_row(row: sqlite3.Row) -> JobHistoryItem:
    try:
        subtitle_formats = json.loads(row["subtitle_formats"])
    except (json.JSONDecodeError, TypeError):
        subtitle_formats = []
    return JobHistoryItem(
        id=row["id"],
        status=row["status"],
        language=row["language"],
        voice=row["voice"],
        source_language=row["source_language"],
        subtitles=bool(row["subtitles"]),
        subtitle_formats=list(subtitle_formats),
        burn_subtitles=bool(row["burn_subtitles"]),
        style=row["style"],
        voiceover=bool(row["voiceover"]),
        source_url=row["source_url"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        error=row["error"],
        progress_count=row["progress_count"],
        deleted=bool(row["deleted"]),
    )


def _status_value(status: JobStatus | str) -> str:
    return status.value if isinstance(status, JobStatus) else str(status)


def _subtitle_formats_from_meta(meta: dict[str, Any]) -> list[str]:
    formats = meta.get("subtitle_formats") or ("srt",)
    if isinstance(formats, str):
        return [fmt.strip() for fmt in formats.split(",") if fmt.strip()]
    return list(formats)


def cleanup_old_jobs(max_age_hours: float) -> int:
    """Delete completed/failed jobs whose meta.json is older than *max_age_hours*.

    Returns the number of jobs deleted. Skips running/queued jobs.
    """
    if max_age_hours <= 0 or not JOBS_DIR.exists():
        return 0

    cutoff = time.time() - max_age_hours * 3600
    deleted = 0

    for job_dir in JOBS_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        meta_file = job_dir / "meta.json"
        if not meta_file.exists():
            continue
        try:
            if meta_file.stat().st_mtime > cutoff:
                continue
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            status = meta.get("status", "")
            if status not in (JobStatus.done, JobStatus.failed, "done", "failed"):
                continue
            shutil.rmtree(job_dir)
            deleted += 1
        except Exception:
            continue

    return deleted


def _atomic_write(path: Path, data: str) -> None:
    """Write data to a file atomically via temp-file + rename."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, data.encode("utf-8"))
        os.close(fd)
        fd = -1
        os.replace(tmp, path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_meta(job_id: str) -> dict[str, Any]:
    """Read meta.json with a single retry in case of a partial-write race."""
    path = _meta_path(job_id)
    for attempt in range(3):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            if attempt < 2:
                time.sleep(0.05)
            else:
                raise
