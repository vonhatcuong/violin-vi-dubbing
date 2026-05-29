import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import app
from api.models import JobResponse, JobStatus


def _job():
    return JobResponse(id="j", status=JobStatus.queued, language="Vietnamese",
                       voice="", source_language="auto-detect", subtitles=True)


class UrlJobPreferCaptionsTests(unittest.TestCase):
    def test_from_url_defaults_prefer_source_captions_true(self):
        with patch("api.routes.jobs.create_job"), \
             patch("api.routes.jobs.submit_url_job") as submit, \
             patch("api.routes.jobs.get_job", return_value=_job()), \
             patch("api.routes.jobs.record_usage"):
            client = TestClient(app)
            res = client.post("/jobs/from-url", json={
                "url": "https://youtu.be/x", "language": "Vietnamese",
                "together_api_key": "k",  # own key → bypass free-trial gate
            })
        self.assertEqual(202, res.status_code)
        params = submit.call_args.args[1]
        self.assertTrue(params["prefer_source_captions"])

    def test_from_url_respects_prefer_source_captions_false(self):
        with patch("api.routes.jobs.create_job"), \
             patch("api.routes.jobs.submit_url_job") as submit, \
             patch("api.routes.jobs.get_job", return_value=_job()), \
             patch("api.routes.jobs.record_usage"):
            client = TestClient(app)
            res = client.post("/jobs/from-url", json={
                "url": "https://youtu.be/x", "language": "Vietnamese",
                "together_api_key": "k", "prefer_source_captions": False,
            })
        self.assertEqual(202, res.status_code)
        params = submit.call_args.args[1]
        self.assertFalse(params["prefer_source_captions"])


if __name__ == "__main__":
    unittest.main()
