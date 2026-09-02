import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api import storage
from api.app import app
from pipeline.extractor import ensure_video_input


class AudioInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_jobs_dir = storage.JOBS_DIR
        storage.JOBS_DIR = Path(self.tmp.name)

    def tearDown(self):
        storage.JOBS_DIR = self.old_jobs_dir
        self.tmp.cleanup()

    def test_upload_route_accepts_audio_files(self):
        with patch("api.routes.jobs.has_free_trial", return_value=True), \
             patch("api.routes.jobs.record_usage"), \
             patch("api.routes.jobs.submit_job") as submit:
            res = TestClient(app).post(
                "/jobs",
                data={
                    "language": "Vietnamese",
                    "subtitles": "true",
                    "subtitle_formats": "srt,txt",
                    "burn_subtitles": "false",
                    "voiceover": "true",
                },
                files={"file": ("clip.mp3", b"fake-mp3", "audio/mpeg")},
            )

        self.assertEqual(202, res.status_code, res.text)
        submit.assert_called_once()

    def test_ensure_video_input_converts_audio_only_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "input_video.mp4"
            probe = MagicMock(stdout='{"streams":[{"codec_type":"audio"}]}')
            with patch("pipeline.extractor.subprocess.run", return_value=probe) as run:
                result = ensure_video_input("input.mp3", str(out))

        self.assertEqual(str(out), result)
        ffmpeg_cmd = run.call_args_list[1].args[0]
        self.assertIn("-f", ffmpeg_cmd)
        self.assertIn("lavfi", ffmpeg_cmd)
        self.assertIn("input.mp3", ffmpeg_cmd)


if __name__ == "__main__":
    unittest.main()
