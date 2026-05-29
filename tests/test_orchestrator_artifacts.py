import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.orchestrator import DubOptions, dub_video
from pipeline.transcriber import Segment


class OrchestratorArtifactTests(unittest.TestCase):
    def test_dub_video_writes_requested_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.mp4"
            srt = Path(tmp) / "out.srt"
            burned = Path(tmp) / "out_subtitled.mp4"
            transcript = Path(tmp) / "transcript.txt"
            tts = Path(tmp) / "seg.wav"
            out.write_bytes(b"video")
            tts.write_bytes(b"wav")

            segments = [Segment(id=0, start=0.0, end=1.0, text="Hello")]
            translated = [Segment(id=0, start=0.0, end=1.0, text="Xin chao")]

            with patch("pipeline.orchestrator.make_translation_client"), \
                 patch("pipeline.orchestrator.make_transcription_client"), \
                 patch("pipeline.orchestrator.extract_audio", return_value=str(Path(tmp) / "audio.wav")), \
                 patch("pipeline.orchestrator.get_video_duration", return_value=1.0), \
                 patch("pipeline.orchestrator.ensure_video_input", return_value="input.mp4"), \
                 patch("pipeline.orchestrator.transcribe", return_value=segments), \
                 patch("pipeline.orchestrator.translate_segments", return_value=translated), \
                 patch("pipeline.orchestrator.synthesize_segments", return_value=[str(tts)]), \
                 patch("pipeline.orchestrator.prepare_merge", return_value=object()), \
                 patch("pipeline.orchestrator.build_gap_chunks"), \
                 patch("pipeline.orchestrator.build_aligned_video", return_value=translated), \
                 patch("pipeline.orchestrator.burn_subtitles") as burn:
                result = dub_video(
                    "input.mp4",
                    str(out),
                    DubOptions(
                        target_language="Vietnamese",
                        subtitles=True,
                        subtitle_formats=("srt", "vtt", "txt"),
                        burn_subtitles=True,
                    ),
                    output_srt_path=str(srt),
                    transcript_path=str(transcript),
                    burned_video_path=str(burned),
                )

            self.assertEqual({"srt", "vtt", "txt"}, set(result.subtitle_paths))
            self.assertTrue(srt.exists())
            self.assertTrue(srt.with_suffix(".vtt").exists())
            self.assertTrue(srt.with_suffix(".txt").exists())
            self.assertEqual("Xin chao\n", transcript.read_text())
            burn.assert_called_once_with(str(out), str(srt), str(burned))
            self.assertEqual(str(transcript), result.transcript_path)
            self.assertEqual(str(burned), result.burned_video_path)


    def test_segments_override_skips_transcription(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.mp4"
            out.write_bytes(b"video")
            tts = Path(tmp) / "seg.wav"
            tts.write_bytes(b"wav")

            override = [Segment(id=0, start=0.0, end=1.0, text="From caption")]
            translated = [Segment(id=0, start=0.0, end=1.0, text="Tu phu de")]

            with patch("pipeline.orchestrator.make_translation_client"), \
                 patch("pipeline.orchestrator.make_transcription_client") as mk_tc, \
                 patch("pipeline.orchestrator.extract_audio") as extract, \
                 patch("pipeline.orchestrator.get_video_duration", return_value=1.0), \
                 patch("pipeline.orchestrator.ensure_video_input", return_value="input.mp4"), \
                 patch("pipeline.orchestrator.transcribe") as transcribe_mock, \
                 patch("pipeline.orchestrator.translate_segments", return_value=translated), \
                 patch("pipeline.orchestrator.synthesize_segments", return_value=[str(tts)]), \
                 patch("pipeline.orchestrator.prepare_merge", return_value=object()), \
                 patch("pipeline.orchestrator.build_gap_chunks"), \
                 patch("pipeline.orchestrator.build_aligned_video", return_value=translated):
                result = dub_video(
                    "input.mp4", str(out),
                    DubOptions(target_language="Vietnamese", subtitles=False),
                    segments_override=override,
                )

            transcribe_mock.assert_not_called()
            extract.assert_not_called()
            mk_tc.assert_not_called()
            self.assertEqual("Tu phu de", result.aligned_segments[0].text)


if __name__ == "__main__":
    unittest.main()
