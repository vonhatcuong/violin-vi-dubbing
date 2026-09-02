import tempfile
import unittest
from pathlib import Path

from api.models import JobStatus
from api import storage


class StorageHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_jobs_dir = storage.JOBS_DIR
        storage.JOBS_DIR = Path(self.tmp.name)

    def tearDown(self):
        storage.JOBS_DIR = self.old_jobs_dir
        self.tmp.cleanup()

    def test_create_update_and_progress_are_mirrored_to_sqlite(self):
        storage.create_job("job-1", {
            "language": "Vietnamese",
            "voice": "vi-VN-NamMinhNeural",
            "source_language": "auto-detect",
            "subtitles": True,
            "subtitle_formats": ("srt", "vtt", "txt"),
            "burn_subtitles": True,
            "style": "standard",
            "voiceover": True,
            "source_url": "https://youtube.com/watch?v=test",
        })
        storage.append_progress("job-1", 1, 5, "Downloading")
        storage.update_status("job-1", JobStatus.done)

        history = storage.list_job_history()

        self.assertEqual(1, len(history))
        row = history[0]
        self.assertEqual("job-1", row.id)
        self.assertEqual(JobStatus.done, row.status)
        self.assertEqual("Vietnamese", row.language)
        self.assertEqual("https://youtube.com/watch?v=test", row.source_url)
        self.assertEqual(["srt", "vtt", "txt"], row.subtitle_formats)
        self.assertTrue(row.burn_subtitles)
        self.assertEqual(1, row.progress_count)

    def test_delete_marks_sqlite_history_deleted_without_losing_row(self):
        storage.create_job("job-2", {
            "language": "Vietnamese",
            "voice": "",
            "source_language": "auto-detect",
            "subtitles": True,
            "style": "standard",
            "voiceover": True,
        })

        self.assertTrue(storage.delete_job("job-2"))
        row = storage.list_job_history()[0]

        self.assertEqual("job-2", row.id)
        self.assertEqual(JobStatus.cancelled, row.status)
        self.assertTrue(row.deleted)


if __name__ == "__main__":
    unittest.main()
