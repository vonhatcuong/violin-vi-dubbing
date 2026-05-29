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
        # end of a word == start of next
        self.assertAlmostEqual(words[1].start, words[0].end, places=3)
        # last word gets +0.30s tail
        self.assertAlmostEqual(words[2].start + 0.30, words[2].end, places=3)


class AlignTests(unittest.TestCase):
    def _w(self, items):
        return [captions._Word(t, s, e) for t, s, e in items]

    def test_chunk_breaks_on_gap(self):
        words = self._w([("a", 0.0, 0.5), ("b", 0.5, 1.0), ("c", 4.0, 4.5)])
        chunks = captions._chunk_words(words, max_words=100, max_gap=2.0)
        self.assertEqual([["a", "b"], ["c"]], [[w.text for w in c] for c in chunks])

    def test_align_splits_sentences_on_terminal_punct(self):
        words = self._w([("a", 0.0, 1.0), ("few", 1.0, 2.0), ("years", 2.0, 3.0),
                         ("hello", 3.0, 4.0), ("there", 4.0, 5.0)])
        punctuated = "A few years. Hello there?"
        segs = captions._align_chunk(words, punctuated)
        self.assertEqual(2, len(segs))
        self.assertEqual("A few years.", segs[0].text)
        self.assertAlmostEqual(0.0, segs[0].start, places=3)
        self.assertAlmostEqual(3.0, segs[0].end, places=3)
        self.assertEqual("Hello there?", segs[1].text)
        self.assertAlmostEqual(3.0, segs[1].start, places=3)
        self.assertAlmostEqual(5.0, segs[1].end, places=3)

    def test_align_returns_none_on_large_token_mismatch(self):
        words = self._w([("a", 0.0, 1.0), ("b", 1.0, 2.0)])
        # LLM returned far more tokens than words → signal fallback
        segs = captions._align_chunk(words, "a b c d e f g h.")
        self.assertIsNone(segs)


if __name__ == "__main__":
    unittest.main()
