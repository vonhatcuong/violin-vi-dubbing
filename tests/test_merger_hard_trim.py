import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline import config as pipeline_config
from pipeline import merger
from pipeline.transcriber import Segment

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def _video(path: Path, seconds: float = 6.0) -> str:
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=25:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], check=True)
    return str(path)


def _tone(path: Path, seconds: float) -> str:
    sr = 44100
    t = np.arange(int(sr * seconds)) / sr
    sf.write(path, (0.5 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), sr)
    return str(path)


def test_hard_trim_keeps_video_length_and_fades_audio(tmp_path, monkeypatch):
    cfg = pipeline_config.load()
    monkeypatch.setitem(cfg["merge_video"], "speed_clamp_min", 1.0)   # no video stretch
    monkeypatch.setitem(cfg["merge_video"], "speed_clamp_max", 1.0)
    monkeypatch.setitem(cfg["merge_video"], "max_audio_speedup", 1.0)  # no atempo
    monkeypatch.setitem(cfg["merge_video"], "max_freeze_s", 0.0)
    monkeypatch.setitem(cfg["merge_video"], "hard_trim", True)
    monkeypatch.setitem(cfg["merge_video"], "trim_fade_ms", 80)
    monkeypatch.setitem(cfg["merge_video"], "workers", 2)

    video = _video(tmp_path / "in.mp4")
    tts = _tone(tmp_path / "tts.wav", seconds=3.0)          # 1 s longer than the 2 s slot
    segs = [Segment(id=0, start=1.0, end=3.0, text="x")]
    out = str(tmp_path / "out.mp4")
    new_segs = merger.build_aligned_video(video, segs, [tts], 6.0, out)

    assert new_segs[0].end - new_segs[0].start == pytest.approx(2.0, abs=0.05)   # no freeze added
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", out], capture_output=True, text=True).stdout)
    assert dur == pytest.approx(6.0, abs=0.15)
    # last 40 ms before the trim point must be quieter than the middle of the clip
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", out, "-ac", "1", "-ar", "44100", str(tmp_path / "out.wav")], check=True)
    audio, sr = sf.read(str(tmp_path / "out.wav"))
    mid = audio[int(sr * 2.0):int(sr * 2.2)]
    tail = audio[int(sr * 2.96):int(sr * 3.0)]
    assert np.abs(tail).mean() < 0.5 * np.abs(mid).mean()
