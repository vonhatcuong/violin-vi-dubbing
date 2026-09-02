import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api import storage
from api.app import app


class JobsHistoryRouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_jobs_dir = storage.JOBS_DIR
        storage.JOBS_DIR = Path(self.tmp.name)

    def tearDown(self):
        storage.JOBS_DIR = self.old_jobs_dir
        self.tmp.cleanup()

    def test_history_endpoint_returns_sqlite_job_rows(self):
        storage.create_job("job-route", {
            "language": "Vietnamese",
            "voice": "",
            "source_language": "auto-detect",
            "subtitles": True,
            "subtitle_formats": ("srt", "txt"),
            "burn_subtitles": False,
            "style": "standard",
            "voiceover": True,
        })

        res = TestClient(app).get("/jobs/history")

        self.assertEqual(200, res.status_code)
        self.assertEqual("job-route", res.json()[0]["id"])
        self.assertEqual(["srt", "txt"], res.json()[0]["subtitle_formats"])


if __name__ == "__main__":
    unittest.main()
