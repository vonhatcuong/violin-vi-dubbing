"""Temp-dir hygiene: the merger and the chunked transcriber must not leave scratch dirs behind —
and the cleanup must never be able to delete anything outside the system temp dir.

Background: a 74-minute lecture left ~2.5 GB of ``vidmerge_*`` .ts intermediates and ~100 MB of
``audiochunk_*`` WAVs per run (a 36-video batch filled a 100 GB disk); an earlier, unguarded fix
resolved a relative fake chunk path to "." and deleted the working directory.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline import config as pipeline_config
from pipeline import merger, transcriber
from pipeline.transcriber import Segment


def _scratch_dirs(prefix: str) -> set[str]:
    return {p.name for p in Path(tempfile.gettempdir()).glob(prefix + "*")}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_build_aligned_video_removes_its_vidmerge_dir(tmp_path):
    pipeline_config.load()
    video = tmp_path / "in.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video),
    ], check=True)
    sr = 44100
    t = np.arange(sr) / sr
    tts = tmp_path / "tts.wav"
    sf.write(tts, (0.3 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), sr)

    before = _scratch_dirs("vidmerge_")
    merger.build_aligned_video(str(video), [Segment(id=0, start=1.0, end=2.0, text="x")], [str(tts)], 3.0,
                               str(tmp_path / "out.mp4"))
    assert (tmp_path / "out.mp4").exists()
    assert _scratch_dirs("vidmerge_") - before == set()


def _fake_chunks(monkeypatch, chunks):
    monkeypatch.setattr(transcriber, "split_audio", lambda audio_path, chunk_seconds=600: chunks)
    monkeypatch.setattr(transcriber, "_transcribe_single",
                        lambda audio_path, client, model: [Segment(id=0, start=0.5, end=1.0, text="hi")])


def test_transcribe_removes_audiochunk_dir(monkeypatch):
    pipeline_config.load()
    chunk_dir = Path(tempfile.mkdtemp(prefix="audiochunk_"))
    (chunk_dir / "chunk_0000.wav").write_bytes(b"")
    (chunk_dir / "chunk_0001.wav").write_bytes(b"")
    _fake_chunks(monkeypatch, [(str(chunk_dir / "chunk_0000.wav"), 0.0), (str(chunk_dir / "chunk_0001.wav"), 600.0)])

    out = transcriber.transcribe("x.wav", client=object())

    assert [s.start for s in out] == [0.5, 600.5]
    assert not chunk_dir.exists()


def test_transcribe_removes_audiochunk_dir_even_when_a_chunk_fails(monkeypatch):
    pipeline_config.load()
    chunk_dir = Path(tempfile.mkdtemp(prefix="audiochunk_"))
    (chunk_dir / "chunk_0000.wav").write_bytes(b"")
    monkeypatch.setattr(transcriber, "split_audio",
                        lambda audio_path, chunk_seconds=600: [(str(chunk_dir / "chunk_0000.wav"), 0.0),
                                                              (str(chunk_dir / "chunk_0001.wav"), 600.0)])

    def boom(audio_path, client, model):
        raise RuntimeError("asr failed")

    monkeypatch.setattr(transcriber, "_transcribe_single", boom)
    with pytest.raises(RuntimeError):
        transcriber.transcribe("x.wav", client=object())
    assert not chunk_dir.exists()


def test_cleanup_never_touches_relative_or_foreign_dirs(monkeypatch, tmp_path):
    """Relative fake chunk paths resolve to the CWD; a caller-owned dir is not ours: both must survive."""
    pipeline_config.load()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "keep.txt").write_text("cwd must survive")
    _fake_chunks(monkeypatch, [("a.wav", 0.0), ("b.wav", 600.0)])
    transcriber.transcribe("x.wav", client=object())
    assert (tmp_path / "keep.txt").read_text() == "cwd must survive"

    foreign = tmp_path / "audiochunk_mine"          # right name, wrong place (not under the system temp dir)
    foreign.mkdir()
    (foreign / "c.wav").write_bytes(b"")
    _fake_chunks(monkeypatch, [(str(foreign / "c.wav"), 0.0), (str(foreign / "d.wav"), 600.0)])
    transcriber.transcribe("x.wav", client=object())
    assert foreign.exists()

    named_wrong = Path(tempfile.mkdtemp(prefix="notours_"))   # right place, wrong name
    (named_wrong / "c.wav").write_bytes(b"")
    _fake_chunks(monkeypatch, [(str(named_wrong / "c.wav"), 0.0), (str(named_wrong / "d.wav"), 600.0)])
    try:
        transcriber.transcribe("x.wav", client=object())
        assert named_wrong.exists()
    finally:
        shutil.rmtree(named_wrong, ignore_errors=True)
    assert os.getcwd() == str(tmp_path)
