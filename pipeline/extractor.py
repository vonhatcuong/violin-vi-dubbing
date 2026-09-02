"""Extract audio from video using ffmpeg."""

import subprocess
import tempfile
import json
from pathlib import Path

import ffmpeg

from .ffmpeg_utils import FFMPEG_EXE, get_duration_video

_OVERLAP_SECONDS = 1  # small overlap to avoid cutting mid-word


def extract_audio(video_path: str, output_path: str | None = None) -> str:
    """Extract audio from video as 16kHz mono WAV — optimal for Whisper."""
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(tempfile.mkdtemp()) / f"{stem}_audio.wav")

    (
        ffmpeg.input(video_path)
        .output(output_path, ar=16000, ac=1, acodec="pcm_s16le")
        .overwrite_output()
        .run(quiet=True, cmd=FFMPEG_EXE)
    )
    return output_path


def has_video_stream(media_path: str) -> bool:
    """Return True when *media_path* has at least one video stream."""
    result = subprocess.run([
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=codec_type",
        "-of", "json",
        media_path,
    ], check=True, capture_output=True, text=True)
    data = json.loads(result.stdout or "{}")
    return any(stream.get("codec_type") == "video" for stream in data.get("streams", []))


def ensure_video_input(input_path: str, output_path: str) -> str:
    """Return a video path suitable for the merger.

    Video inputs are returned unchanged. Audio-only inputs are wrapped in a
    black MP4 with the original audio track so the existing video merge and
    subtitle burn stages can run unchanged.
    """
    if has_video_stream(input_path):
        return input_path

    subprocess.run([
        FFMPEG_EXE,
        "-f", "lavfi", "-i", "color=c=black:s=1280x720:r=30",
        "-i", input_path,
        "-shortest",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        "-y", output_path,
    ], check=True, capture_output=True)
    return output_path


def split_audio(audio_path: str, output_dir: str | None = None,
                chunk_seconds: float = 600) -> list[tuple[str, float]]:
    """Split a WAV file into chunks of *chunk_seconds*.

    Returns a list of (chunk_path, offset_seconds) tuples.  If the file is
    shorter than one chunk, returns a single-element list with offset 0.
    """
    duration = get_duration_video(audio_path)
    if duration <= chunk_seconds:
        return [(audio_path, 0.0)]

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="audiochunk_")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    chunks: list[tuple[str, float]] = []
    offset = 0.0
    idx = 0
    while offset < duration:
        chunk_path = str(out / f"chunk_{idx:04d}.wav")
        length = min(chunk_seconds + _OVERLAP_SECONDS, duration - offset)
        subprocess.run([
            FFMPEG_EXE,
            "-ss", str(offset),
            "-t", str(length),
            "-i", audio_path,
            "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-y", chunk_path,
        ], check=True, capture_output=True)
        chunks.append((chunk_path, offset))
        offset += chunk_seconds
        idx += 1

    return chunks


def get_video_duration(video_path: str) -> float:
    return get_duration_video(video_path)
