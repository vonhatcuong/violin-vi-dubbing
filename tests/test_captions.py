import unittest
from unittest.mock import MagicMock, patch

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
        info = {"automatic_captions": {"vi": _track("TRANS"), "aa-en": _track("X")}}
        self.assertIsNone(captions._select_track(info, "en"))

    def test_manual_variant_match(self):
        info = {"subtitles": {"en-US": _track("MANUS")}}
        t = captions._select_track(info, "en")
        self.assertEqual(("manual", "en-US", "MANUS"), (t.kind, t.lang, t.url))

    def test_none_when_no_src_and_no_manual(self):
        info = {"automatic_captions": {"en": _track("AUTO")}}
        self.assertIsNone(captions._select_track(info, None))


class ParseManualTests(unittest.TestCase):
    def test_cue_events_become_segments_with_timestamps(self):
        data = {"events": [
            {"tStartMs": 13240, "dDurationMs": 2560,
             "segs": [{"utf8": "A few years ago,\nI broke in."}]},
            {"tStartMs": 16880, "dDurationMs": 1216,
             "segs": [{"utf8": "I had just driven home,"}]},
            {"tStartMs": 9999, "segs": [{"utf8": "  \n "}]},  # blank → dropped
        ]}
        segs = captions._parse_manual(data)
        self.assertEqual(2, len(segs))
        self.assertEqual("A few years ago, I broke in.", segs[0].text)
        self.assertAlmostEqual(13.24, segs[0].start, places=2)
        self.assertAlmostEqual(15.80, segs[0].end, places=2)
        self.assertEqual([0, 1], [s.id for s in segs])


class ParseAutoWordsTests(unittest.TestCase):
    def test_word_level_with_offsets_and_dedup(self):
        data = {"events": [
            {"tStartMs": 1000, "dDurationMs": 2000, "segs": [
                {"utf8": "a", "tOffsetMs": 0},
                {"utf8": " few", "tOffsetMs": 200},
            ]},
            {"tStartMs": 1500, "segs": [{"utf8": "\n"}]},  # noise → dropped
            {"tStartMs": 1180, "dDurationMs": 1500, "segs": [
                {"utf8": "few", "tOffsetMs": 20},   # rolling dup of "few"@1.2 → dropped
                {"utf8": " years", "tOffsetMs": 300},
            ]},
        ]}
        words = captions._parse_auto_words(data)
        self.assertEqual(["a", "few", "years"], [w.text for w in words])
        self.assertAlmostEqual(1.0, words[0].start, places=3)
        self.assertAlmostEqual(1.2, words[1].start, places=3)
        self.assertAlmostEqual(words[1].start, words[0].end, places=3)
        self.assertAlmostEqual(words[2].start + 0.30, words[2].end, places=3)


class SegmentAutoWordsTests(unittest.TestCase):
    def _w(self, items):
        return [captions._Word(t, s, e) for t, s, e in items]

    def test_breaks_on_pause_capitalizes_and_periods(self):
        words = self._w([("hello", 0.0, 0.4), ("world", 0.4, 0.8),   # gap 0.7 > 0.5 → break
                         ("how", 1.5, 1.8), ("are", 1.8, 2.0), ("you", 2.0, 2.3)])
        segs = captions._segment_auto_words(words, max_pause=0.5, max_words=20)
        self.assertEqual(["Hello world.", "How are you."], [s.text for s in segs])
        self.assertAlmostEqual(0.0, segs[0].start, places=3)
        self.assertAlmostEqual(0.8, segs[0].end, places=3)
        self.assertAlmostEqual(1.5, segs[1].start, places=3)
        self.assertAlmostEqual(2.3, segs[1].end, places=3)

    def test_caps_by_max_words(self):
        # gaps are all 0.5s (no pause break with max_pause=2.0); cap at 2 words
        words = self._w([(f"w{i}", i * 1.0, i * 1.0 + 0.5) for i in range(5)])
        segs = captions._segment_auto_words(words, max_pause=2.0, max_words=2)
        self.assertEqual([2, 2, 1], [len(s.text.split()) for s in segs])
        self.assertTrue(all(s.text.endswith(".") for s in segs))

    def test_keeps_existing_terminal_punctuation(self):
        words = self._w([("ok", 0.0, 0.5), ("done!", 0.5, 1.0)])
        segs = captions._segment_auto_words(words, max_pause=2.0, max_words=20)
        self.assertEqual(["Ok done!"], [s.text for s in segs])


class FetchSourceCaptionsTests(unittest.TestCase):
    def _ydl_with(self, info):
        fake = MagicMock()
        fake.__enter__.return_value.extract_info.return_value = info
        return fake

    def test_manual_path_returns_segments(self):
        info = {"language": "en",
                "subtitles": {"en": [{"ext": "json3", "url": "MAN_URL"}]},
                "automatic_captions": {}}
        manual_json = {"events": [
            {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "Hello there."}]},
        ]}
        with patch("pipeline.captions._open_ydl", return_value=self._ydl_with(info)), \
             patch("pipeline.captions._download_json3", return_value=manual_json):
            segs = captions.fetch_source_captions("https://x/y", "English")
        self.assertEqual(["Hello there."], [s.text for s in segs])

    def test_auto_path_segments_without_llm(self):
        info = {"language": "en", "subtitles": {},
                "automatic_captions": {"en": [{"ext": "json3", "url": "AUTO_URL"}]}}
        auto_json = {"events": [
            {"tStartMs": 0, "segs": [{"utf8": "hello", "tOffsetMs": 0},
                                     {"utf8": " world", "tOffsetMs": 400}]},
            {"tStartMs": 2000, "segs": [{"utf8": "bye", "tOffsetMs": 0}]},  # 1.2s gap → new seg
        ]}
        with patch("pipeline.captions._open_ydl", return_value=self._ydl_with(info)), \
             patch("pipeline.captions._download_json3", return_value=auto_json):
            segs = captions.fetch_source_captions("https://x/y", "auto-detect")
        self.assertEqual(["Hello world.", "Bye."], [s.text for s in segs])

    def test_returns_none_when_no_track(self):
        info = {"language": "en", "subtitles": {}, "automatic_captions": {}}
        with patch("pipeline.captions._open_ydl", return_value=self._ydl_with(info)):
            self.assertIsNone(captions.fetch_source_captions("https://x/y", "English"))

    def test_returns_none_on_extract_error(self):
        with patch("pipeline.captions._open_ydl", side_effect=RuntimeError("net")):
            self.assertIsNone(captions.fetch_source_captions("https://x/y", "English"))


if __name__ == "__main__":
    unittest.main()
