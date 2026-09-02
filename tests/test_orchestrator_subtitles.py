import tempfile
from pathlib import Path
from unittest.mock import patch

from pipeline import config as pipeline_config
from pipeline.orchestrator import DubOptions, dub_video
from pipeline.transcriber import Segment


def test_source_subtitles_use_english_sentences_retimed_to_output():
    pipeline_config.load()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.mp4"; out.write_bytes(b"video")
        srt = Path(tmp) / "out.srt"
        tts = Path(tmp) / "seg.wav"; tts.write_bytes(b"wav")
        raw = [Segment(id=0, start=0.0, end=2.0, text="Hello there."),
               Segment(id=1, start=2.5, end=4.0, text="Second one.")]
        translated = [Segment(id=0, start=0.0, end=2.0, text="Xin chào.", source_text="Hello there."),
                      Segment(id=1, start=2.5, end=4.0, text="Câu hai.", source_text="Second one.")]
        # merger stretched unit 0 by 1 s; the 0.5 s gap is copied, unit 1 shifts by +1
        aligned = [Segment(id=0, start=0.0, end=3.0, text="Xin chào.", source_text="Hello there."),
                   Segment(id=1, start=3.5, end=5.0, text="Câu hai.", source_text="Second one.")]

        with patch("pipeline.orchestrator.make_translation_client"), \
             patch("pipeline.orchestrator.make_transcription_client"), \
             patch("pipeline.orchestrator.extract_audio", return_value=str(Path(tmp) / "a.wav")), \
             patch("pipeline.orchestrator.get_video_duration", return_value=6.0), \
             patch("pipeline.orchestrator.ensure_video_input", return_value="input.mp4"), \
             patch("pipeline.orchestrator.transcribe", return_value=raw), \
             patch("pipeline.orchestrator.translate_segments", return_value=translated), \
             patch("pipeline.orchestrator.synthesize_segments", return_value=[str(tts), str(tts)]), \
             patch("pipeline.orchestrator.prepare_merge", return_value=object()), \
             patch("pipeline.orchestrator.build_gap_chunks"), \
             patch("pipeline.orchestrator.build_aligned_video", return_value=aligned):
            result = dub_video("input.mp4", str(out),
                               DubOptions(target_language="Vietnamese", subtitles=True, subtitle_lang="source"),
                               output_srt_path=str(srt))

        text = srt.read_text(encoding="utf-8")
        assert "Hello there." in text and "Second one." in text
        assert "Xin chào" not in text
        assert "00:00:00,000 --> 00:00:03,000" in text
        assert "00:00:03,500 --> 00:00:05,000" in text
        # transcript stays Vietnamese
        assert "Xin chào." in Path(result.transcript_path).read_text(encoding="utf-8")


def test_target_subtitles_remain_default():
    pipeline_config.load()
    assert DubOptions(target_language="Vietnamese").subtitle_lang is None
    assert pipeline_config.get()["subtitles"]["language"] == "target"
