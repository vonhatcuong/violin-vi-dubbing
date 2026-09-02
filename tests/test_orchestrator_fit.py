import tempfile
from pathlib import Path
from unittest.mock import patch

from pipeline import config as pipeline_config
from pipeline.fitter import DubUnit
from pipeline.orchestrator import DubOptions, dub_video
from pipeline.transcriber import Segment


def test_fit_path_replaces_synthesize_and_keeps_sentence_units(monkeypatch):
    cfg = pipeline_config.load()
    monkeypatch.setitem(cfg["fit"], "enabled", True)
    monkeypatch.setitem(cfg["models"], "tts", {"provider": "vieneu", "model": "vieneu-v3-turbo"})

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.mp4"
        out.write_bytes(b"video")
        tts = Path(tmp) / "seg.wav"
        tts.write_bytes(b"wav")
        segments = [Segment(id=0, start=0.0, end=2.0, text="Hello. World.")]
        translated = [Segment(id=0, start=0.0, end=2.0, text="Chào. Thế giới.", source_text="Hello. World.")]

        def fake_fit_audio(units, synth, out_dir, fcfg, synth_batch=None):
            for u in units:
                u.tts_path, u.tts_dur = str(tts), 1.0

        with patch("pipeline.orchestrator.make_translation_client"), \
             patch("pipeline.orchestrator.make_transcription_client"), \
             patch("pipeline.orchestrator.extract_audio", return_value=str(Path(tmp) / "a.wav")), \
             patch("pipeline.orchestrator.get_video_duration", return_value=5.0), \
             patch("pipeline.orchestrator.ensure_video_input", return_value="input.mp4"), \
             patch("pipeline.orchestrator.transcribe", return_value=segments), \
             patch("pipeline.orchestrator.translate_segments", return_value=translated) as tr, \
             patch("pipeline.orchestrator.synthesize_segments") as synth_segments, \
             patch("pipeline.orchestrator.make_synthesizer", return_value=lambda *a: str(tts)), \
             patch("pipeline.orchestrator.make_batch_synthesizer", return_value=None), \
             patch("pipeline.orchestrator.fitter.fit_text") as fit_text, \
             patch("pipeline.orchestrator.fitter.fit_audio", side_effect=fake_fit_audio) as fit_audio, \
             patch("pipeline.orchestrator._resolve_voice", return_value="nam-1"), \
             patch("pipeline.orchestrator.prepare_merge", return_value=object()), \
             patch("pipeline.orchestrator.build_gap_chunks"), \
             patch("pipeline.orchestrator.build_aligned_video", side_effect=lambda v, segs, paths, *a, **k: segs):
            result = dub_video("input.mp4", str(out), DubOptions(target_language="Vietnamese", subtitles=False))

        synth_segments.assert_not_called()
        fit_text.assert_called_once()
        fit_audio.assert_called_once()
        assert tr.call_args.kwargs["budgets"] is not None
        # sentence units are NOT re-split after translation on the fit path
        assert [s.text for s in result.aligned_segments] == ["Chào. Thế giới."]
        assert (out.with_suffix(".fit.units.json")).exists()
        assert (out.with_suffix(".fitted.segments.json")).exists()


def test_fit_pipelined_path_skips_translate_segments_and_writes_artifacts(monkeypatch):
    cfg = pipeline_config.load()
    monkeypatch.setitem(cfg["fit"], "enabled", True)
    monkeypatch.setitem(cfg["fit"], "pipelined", True)
    monkeypatch.setitem(cfg["models"], "tts", {"provider": "vieneu", "model": "vieneu-v3-turbo"})

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.mp4"
        out.write_bytes(b"video")
        tts = Path(tmp) / "seg.wav"
        tts.write_bytes(b"wav")
        segments = [Segment(id=0, start=0.0, end=2.0, text="Hello. World.")]
        translated = [Segment(id=0, start=0.0, end=2.0, text="Chào. Thế giới.", source_text="Hello. World.")]
        units = [DubUnit(seg_id=0, speaker="SPEAKER_00", voice="nam-1", source_text="Hello. World.",
                          text="Chào. Thế giới.", start=0.0, end=2.0, slot_end=2.6,
                          tts_path=str(tts), tts_dur=1.0)]

        with patch("pipeline.orchestrator.make_translation_client"), \
             patch("pipeline.orchestrator.make_transcription_client"), \
             patch("pipeline.orchestrator.extract_audio", return_value=str(Path(tmp) / "a.wav")), \
             patch("pipeline.orchestrator.get_video_duration", return_value=5.0), \
             patch("pipeline.orchestrator.ensure_video_input", return_value="input.mp4"), \
             patch("pipeline.orchestrator.transcribe", return_value=segments), \
             patch("pipeline.orchestrator.translate_segments") as tr, \
             patch("pipeline.orchestrator.synthesize_segments") as synth_segments, \
             patch("pipeline.orchestrator.make_synthesizer", return_value=lambda *a: str(tts)), \
             patch("pipeline.orchestrator.make_batch_synthesizer", return_value=None), \
             patch("pipeline.orchestrator.fitter.run_pipelined", return_value=(translated, units)) as run_pipelined, \
             patch("pipeline.orchestrator._resolve_voice", return_value="nam-1"), \
             patch("pipeline.orchestrator.prepare_merge", return_value=object()), \
             patch("pipeline.orchestrator.build_gap_chunks"), \
             patch("pipeline.orchestrator.build_aligned_video", side_effect=lambda v, segs, paths, *a, **k: segs):
            result = dub_video("input.mp4", str(out), DubOptions(target_language="Vietnamese", subtitles=False))

        tr.assert_not_called()
        run_pipelined.assert_called_once()
        synth_segments.assert_not_called()
        assert [s.text for s in result.aligned_segments] == ["Chào. Thế giới."]
        assert (out.with_suffix(".fit.units.json")).exists()
        assert (out.with_suffix(".fitted.segments.json")).exists()
