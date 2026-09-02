import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline import config as pipeline_config
from pipeline.orchestrator import DubOptions, dub_video
from pipeline.transcriber import Segment


def test_dub_options_rejects_invalid_speakers_value():
    with pytest.raises(ValueError):
        DubOptions(target_language="Vietnamese", speakers="two")


def test_speakers_default_off_skips_diarizer_and_writes_no_voices_json():
    pipeline_config.load()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.mp4"
        out.write_bytes(b"video")
        tts = Path(tmp) / "seg.wav"
        tts.write_bytes(b"wav")
        segments = [Segment(id=0, start=0.0, end=1.0, text="Hello")]
        translated = [Segment(id=0, start=0.0, end=1.0, text="Xin chao")]

        with patch("pipeline.orchestrator.make_translation_client"), \
             patch("pipeline.orchestrator.make_transcription_client"), \
             patch("pipeline.orchestrator.extract_audio", return_value=str(Path(tmp) / "a.wav")), \
             patch("pipeline.orchestrator.get_video_duration", return_value=1.0), \
             patch("pipeline.orchestrator.ensure_video_input", return_value="input.mp4"), \
             patch("pipeline.orchestrator.transcribe", return_value=segments), \
             patch("pipeline.orchestrator.diarizer.label_segments") as dl, \
             patch("pipeline.orchestrator.translate_segments", return_value=translated), \
             patch("pipeline.orchestrator.synthesize_segments", return_value=[str(tts)]), \
             patch("pipeline.orchestrator.prepare_merge", return_value=object()), \
             patch("pipeline.orchestrator.build_gap_chunks"), \
             patch("pipeline.orchestrator.build_aligned_video", return_value=translated):
            dub_video("input.mp4", str(out), DubOptions(target_language="Vietnamese", subtitles=False))

        dl.assert_not_called()
        assert not out.with_suffix(".voices.json").exists()
        assert not out.with_suffix(".diarized.segments.json").exists()


def test_speakers_auto_diarizes_and_synthesize_segments_gets_voice_map():
    pipeline_config.load()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.mp4"
        out.write_bytes(b"video")
        tts = Path(tmp) / "seg.wav"
        tts.write_bytes(b"wav")
        segments = [
            Segment(id=0, start=0.0, end=1.0, text="Hello"),
            Segment(id=1, start=1.5, end=2.5, text="Hi"),
        ]
        translated = [
            Segment(id=0, start=0.0, end=1.0, text="Xin chao", speaker="SPEAKER_00"),
            Segment(id=1, start=1.5, end=2.5, text="Chao", speaker="SPEAKER_01"),
        ]

        with patch("pipeline.orchestrator.make_translation_client"), \
             patch("pipeline.orchestrator.make_transcription_client"), \
             patch("pipeline.orchestrator.extract_audio", return_value=str(Path(tmp) / "a.wav")), \
             patch("pipeline.orchestrator.get_video_duration", return_value=3.0), \
             patch("pipeline.orchestrator.ensure_video_input", return_value="input.mp4"), \
             patch("pipeline.orchestrator.transcribe", return_value=segments), \
             patch("pipeline.orchestrator.diarizer.label_segments",
                   return_value=["SPEAKER_00", "SPEAKER_01"]) as dl, \
             patch("pipeline.orchestrator.guess_genders", return_value={}) as gg, \
             patch("pipeline.orchestrator.translate_segments", return_value=translated), \
             patch("pipeline.orchestrator.synthesize_segments",
                   return_value=[str(tts), str(tts)]) as synth_segments, \
             patch("pipeline.orchestrator.prepare_merge", return_value=object()), \
             patch("pipeline.orchestrator.build_gap_chunks"), \
             patch("pipeline.orchestrator.build_aligned_video", return_value=translated):
            dub_video(
                "input.mp4", str(out),
                DubOptions(target_language="Vietnamese", subtitles=False, speakers="auto"),
            )

        dl.assert_called_once()
        assert dl.call_args.kwargs["num_speakers"] is None
        gg.assert_not_called()  # voices.gender_detect defaults to false

        voice_map = synth_segments.call_args.kwargs["voice_map"]
        assert voice_map["SPEAKER_00"] != voice_map["SPEAKER_01"]

        voices_json = out.with_suffix(".voices.json")
        assert voices_json.exists()
        data = json.loads(voices_json.read_text(encoding="utf-8"))
        assert data == voice_map
        assert out.with_suffix(".diarized.segments.json").exists()


def test_explicit_voice_map_overrides_assigned_voice():
    pipeline_config.load()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.mp4"
        out.write_bytes(b"video")
        tts = Path(tmp) / "seg.wav"
        tts.write_bytes(b"wav")
        segments = [
            Segment(id=0, start=0.0, end=1.0, text="Hello"),
            Segment(id=1, start=1.5, end=2.5, text="Hi"),
        ]
        translated = [
            Segment(id=0, start=0.0, end=1.0, text="Xin chao", speaker="SPEAKER_00"),
            Segment(id=1, start=1.5, end=2.5, text="Chao", speaker="SPEAKER_01"),
        ]

        with patch("pipeline.orchestrator.make_translation_client"), \
             patch("pipeline.orchestrator.make_transcription_client"), \
             patch("pipeline.orchestrator.extract_audio", return_value=str(Path(tmp) / "a.wav")), \
             patch("pipeline.orchestrator.get_video_duration", return_value=3.0), \
             patch("pipeline.orchestrator.ensure_video_input", return_value="input.mp4"), \
             patch("pipeline.orchestrator.transcribe", return_value=segments), \
             patch("pipeline.orchestrator.diarizer.label_segments", return_value=["SPEAKER_00", "SPEAKER_01"]), \
             patch("pipeline.orchestrator.guess_genders", return_value={}), \
             patch("pipeline.orchestrator.translate_segments", return_value=translated), \
             patch("pipeline.orchestrator.synthesize_segments",
                   return_value=[str(tts), str(tts)]) as synth_segments, \
             patch("pipeline.orchestrator.prepare_merge", return_value=object()), \
             patch("pipeline.orchestrator.build_gap_chunks"), \
             patch("pipeline.orchestrator.build_aligned_video", return_value=translated):
            dub_video(
                "input.mp4", str(out),
                DubOptions(target_language="Vietnamese", subtitles=False, speakers="auto",
                           voice_map={"SPEAKER_00": "Adam"}),
            )

        voice_map = synth_segments.call_args.kwargs["voice_map"]
        assert voice_map["SPEAKER_00"] == "Adam"


def test_speakers_fixed_count_threads_voice_map_into_build_units(monkeypatch):
    cfg = pipeline_config.load()
    monkeypatch.setitem(cfg["fit"], "enabled", True)
    monkeypatch.setitem(cfg["models"], "tts", {"provider": "vieneu", "model": "vieneu-v3-turbo"})

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.mp4"
        out.write_bytes(b"video")
        tts = Path(tmp) / "seg.wav"
        tts.write_bytes(b"wav")
        segments = [
            Segment(id=0, start=0.0, end=2.0, text="Hello."),
            Segment(id=1, start=3.0, end=5.0, text="Hi there."),
        ]
        translated = [
            Segment(id=0, start=0.0, end=2.0, text="Chao.", source_text="Hello.", speaker="SPEAKER_00"),
            Segment(id=1, start=3.0, end=5.0, text="Chao ban.", source_text="Hi there.", speaker="SPEAKER_01"),
        ]

        captured_voices: dict[str, str] = {}

        def fake_fit_audio(units, synth, out_dir, fcfg, synth_batch=None):
            for u in units:
                u.tts_path, u.tts_dur = str(tts), 1.0
                captured_voices[u.speaker] = u.voice

        with patch("pipeline.orchestrator.make_translation_client"), \
             patch("pipeline.orchestrator.make_transcription_client"), \
             patch("pipeline.orchestrator.extract_audio", return_value=str(Path(tmp) / "a.wav")), \
             patch("pipeline.orchestrator.get_video_duration", return_value=6.0), \
             patch("pipeline.orchestrator.ensure_video_input", return_value="input.mp4"), \
             patch("pipeline.orchestrator.transcribe", return_value=segments), \
             patch("pipeline.orchestrator.diarizer.label_segments",
                   return_value=["SPEAKER_00", "SPEAKER_01"]) as dl, \
             patch("pipeline.orchestrator.guess_genders", return_value={}), \
             patch("pipeline.orchestrator.translate_segments", return_value=translated), \
             patch("pipeline.orchestrator.synthesize_segments") as synth_segments, \
             patch("pipeline.orchestrator.make_synthesizer", return_value=lambda *a: str(tts)), \
             patch("pipeline.orchestrator.make_batch_synthesizer", return_value=None), \
             patch("pipeline.orchestrator.fitter.fit_text"), \
             patch("pipeline.orchestrator.fitter.fit_audio", side_effect=fake_fit_audio), \
             patch("pipeline.orchestrator.prepare_merge", return_value=object()), \
             patch("pipeline.orchestrator.build_gap_chunks"), \
             patch("pipeline.orchestrator.build_aligned_video", side_effect=lambda v, segs, paths, *a, **k: segs):
            dub_video(
                "input.mp4", str(out),
                DubOptions(target_language="Vietnamese", subtitles=False, speakers="2"),
            )

        dl.assert_called_once()
        assert dl.call_args.kwargs["num_speakers"] == 2
        synth_segments.assert_not_called()
        assert len(captured_voices) == 2
        assert len(set(captured_voices.values())) == 2
        assert out.with_suffix(".voices.json").exists()
