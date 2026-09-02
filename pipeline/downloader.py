"""URL media download helpers."""

from __future__ import annotations

from pathlib import Path


def is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def download_url_to_file(
    url: str,
    output_dir: str | Path,
    *,
    max_duration_seconds: int = 0,
    max_file_size_mb: int = 0,
) -> Path:
    """Download a single media URL with yt-dlp and return the local input path."""
    import yt_dlp

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "input.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/bestaudio/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if max_file_size_mb > 0:
        ydl_opts["max_filesize"] = max_file_size_mb * 1024 * 1024

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        duration = info.get("duration", 0)
        if max_duration_seconds > 0 and duration and duration > max_duration_seconds:
            limit_min = max_duration_seconds // 60
            raise ValueError(f"Media too long ({duration // 60} min). Max {limit_min} min.")

    for path in out_dir.glob("input.*"):
        return path
    raise FileNotFoundError("yt-dlp download succeeded but no input file found.")
