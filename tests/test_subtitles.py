import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.merger import burn_subtitles, generate_subtitle_files, generate_transcript
from pipeline.transcriber import Segment


class SubtitleExportTests(unittest.TestCase):
    def test_generate_subtitle_files_writes_srt_vtt_and_txt(self):
        segments = [
            Segment(id=0, start=1.25, end=3.5, text="Xin chao", speaker="SPEAKER_00"),
            Segment(id=1, start=4.0, end=5.125, text="Tam biet", speaker="SPEAKER_00"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_subtitle_files(
                segments,
                str(Path(tmp) / "output.srt"),
                formats=("srt", "vtt", "txt"),
            )

            self.assertEqual({"srt", "vtt", "txt"}, set(paths))
            self.assertIn("00:00:01,250 --> 00:00:03,500", Path(paths["srt"]).read_text())
            self.assertTrue(Path(paths["vtt"]).read_text().startswith("WEBVTT\n\n"))
            self.assertIn("00:00:01.250 --> 00:00:03.500", Path(paths["vtt"]).read_text())
            self.assertEqual(
                "[00:00:01.250] Xin chao\n[00:00:04.000] Tam biet\n",
                Path(paths["txt"]).read_text(),
            )

    def test_burn_subtitles_invokes_ffmpeg_with_subtitles_filter(self):
        with patch("pipeline.merger.subprocess.run") as run:
            burn_subtitles("input.mp4", "captions.srt", "burned.mp4")

        cmd = run.call_args.args[0]
        self.assertIn("-vf", cmd)
        self.assertIn("subtitles=", cmd[cmd.index("-vf") + 1])
        self.assertEqual("burned.mp4", cmd[-1])

    def test_generate_transcript_writes_plain_text_without_timestamps(self):
        segments = [
            Segment(id=0, start=1.25, end=3.5, text="Xin chao", speaker="SPEAKER_00"),
            Segment(id=1, start=4.0, end=5.125, text="Tam biet", speaker="SPEAKER_00"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = generate_transcript(segments, str(Path(tmp) / "transcript.txt"))

            self.assertEqual("Xin chao\nTam biet\n", Path(path).read_text())


if __name__ == "__main__":
    unittest.main()
