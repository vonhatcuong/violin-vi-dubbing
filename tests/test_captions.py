import unittest

from pipeline import captions


def _track(url, ext="json3"):
    return [{"ext": ext, "url": url}]


class SelectTrackTests(unittest.TestCase):
    def test_source_lang_from_explicit_language(self):
        self.assertEqual("en", captions._source_lang_code({}, "English"))

    def test_source_lang_from_info_when_autodetect(self):
        self.assertEqual("en", captions._source_lang_code({"language": "en-US"}, "auto-detect"))

    def test_prefers_manual_exact_over_auto(self):
        info = {
            "subtitles": {"en": _track("MAN")},
            "automatic_captions": {"en": _track("AUTO")},
        }
        t = captions._select_track(info, "en")
        self.assertEqual(("manual", "en", "MAN"), (t.kind, t.lang, t.url))

    def test_uses_auto_asr_when_no_manual(self):
        info = {"automatic_captions": {"en": _track("AUTO"), "vi": _track("TRANS")}}
        t = captions._select_track(info, "en")
        self.assertEqual(("auto", "en", "AUTO"), (t.kind, t.lang, t.url))

    def test_rejects_translated_auto_track(self):
        # only translated tracks exist (target=vi / odd key), source=en → no ASR original
        info = {"automatic_captions": {"vi": _track("TRANS"), "aa-en": _track("X")}}
        self.assertIsNone(captions._select_track(info, "en"))

    def test_manual_variant_match(self):
        info = {"subtitles": {"en-US": _track("MANUS")}}
        t = captions._select_track(info, "en")
        self.assertEqual(("manual", "en-US", "MANUS"), (t.kind, t.lang, t.url))

    def test_none_when_no_src_and_no_manual(self):
        info = {"automatic_captions": {"en": _track("AUTO")}}
        self.assertIsNone(captions._select_track(info, None))


if __name__ == "__main__":
    unittest.main()
