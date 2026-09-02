import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api import storage
from api.app import app
from api.models import JobStatus


class TranscriptRouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_jobs_dir = storage.JOBS_DIR
        storage.JOBS_DIR = Path(self.tmp.name)

    def tearDown(self):
        storage.JOBS_DIR = self.old_jobs_dir
        self.tmp.cleanup()

    def test_transcript_endpoint_serves_plain_transcript(self):
        storage.create_job("job-transcript", {
            "language": "Vietnamese",
            "voice": "",
            "source_language": "auto-detect",
            "subtitles": True,
            "style": "standard",
            "voiceover": True,
        })
        storage.transcript_path("job-transcript").write_text("Xin chao\n", encoding="utf-8")
        storage.update_status("job-transcript", JobStatus.done)

        res = TestClient(app).get("/jobs/job-transcript/transcript")

        self.assertEqual(200, res.status_code)
        self.assertEqual("Xin chao\n", res.text)


if __name__ == "__main__":
    unittest.main()
