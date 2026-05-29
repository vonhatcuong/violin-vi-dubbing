import unittest
from pathlib import Path
from unittest.mock import patch

from api import worker


class WorkerUrlCaptionsTests(unittest.TestCase):
    def test_url_job_passes_caption_segments_to_run_job(self):
        from pipeline.transcriber import Segment
        caps = [Segment(id=0, start=0.0, end=1.0, text="cap")]
        params = {"language": "Vietnamese", "voice": "", "source_language": "auto-detect",
                  "subtitles": True, "style": "standard", "voiceover": True,
                  "prefer_source_captions": True}
        with patch("api.worker.update_status"), \
             patch("api.worker.append_progress"), \
             patch("api.worker._download_url", return_value=Path("/tmp/input.mp4")), \
             patch("api.worker.fetch_source_captions", return_value=caps) as fetch, \
             patch("api.worker._run_job") as run_job:
            worker._run_url_job("job1", params, "https://youtu.be/x")
        fetch.assert_called_once()
        self.assertEqual(caps, run_job.call_args.kwargs["segments_override"])

    def test_url_job_skips_caption_when_disabled(self):
        params = {"language": "Vietnamese", "voice": "", "source_language": "auto-detect",
                  "subtitles": True, "style": "standard", "voiceover": True,
                  "prefer_source_captions": False}
        with patch("api.worker.update_status"), \
             patch("api.worker.append_progress"), \
             patch("api.worker._download_url", return_value=Path("/tmp/input.mp4")), \
             patch("api.worker.fetch_source_captions") as fetch, \
             patch("api.worker._run_job") as run_job:
            worker._run_url_job("job2", params, "https://youtu.be/x")
        fetch.assert_not_called()
        self.assertIsNone(run_job.call_args.kwargs["segments_override"])


if __name__ == "__main__":
    unittest.main()
