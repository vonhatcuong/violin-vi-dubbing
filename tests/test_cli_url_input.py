import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class CliUrlInputTests(unittest.TestCase):
    def test_translate_video_downloads_url_before_running_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            downloaded = Path(tmp) / "input.mp4"
            downloaded.write_bytes(b"video")

            with patch("main.download_url_to_file", return_value=downloaded) as download, \
                 patch("main.fetch_source_captions", return_value=None), \
                 patch("main.dub_video") as dub, \
                 patch("main.resolve_style", return_value=type("Style", (), {
                     "name": "standard",
                     "description": "",
                 })()):
                dub.return_value = type("Result", (), {
                    "original_audio_path": None,
                    "subtitle_paths": {},
                    "burned_video_path": None,
                    "transcript_path": None,
                    "output_video_path": "out.mp4",
                    "cost_tracker": type("Tracker", (), {"print_summary": lambda self: None})(),
                    "steps": [],
                })()

                main.translate_video(
                    "https://www.youtube.com/watch?v=test",
                    "out.mp4",
                    "Vietnamese",
                )

            download.assert_called_once()
            self.assertEqual(str(downloaded), dub.call_args.args[0])

    def test_url_fetches_source_captions_and_passes_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            downloaded = Path(tmp) / "input.mp4"
            downloaded.write_bytes(b"video")
            from pipeline.transcriber import Segment
            caps = [Segment(id=0, start=0.0, end=1.0, text="From caption")]

            with patch("main.download_url_to_file", return_value=downloaded), \
                 patch("main.fetch_source_captions", return_value=caps) as fetch, \
                 patch("main.dub_video") as dub, \
                 patch("main.resolve_style", return_value=type("Style", (), {
                     "name": "standard", "description": ""})()):
                dub.return_value = type("Result", (), {
                    "original_audio_path": None, "subtitle_paths": {},
                    "burned_video_path": None, "transcript_path": None,
                    "output_video_path": "out.mp4",
                    "cost_tracker": type("T", (), {"print_summary": lambda self: None})(),
                    "steps": [],
                })()
                main.translate_video(
                    "https://www.youtube.com/watch?v=test", "out.mp4", "Vietnamese",
                )
            fetch.assert_called_once()
            self.assertEqual(caps, dub.call_args.kwargs["segments_override"])


if __name__ == "__main__":
    unittest.main()
