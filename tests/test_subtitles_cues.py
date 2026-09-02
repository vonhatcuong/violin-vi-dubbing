import pytest

from pipeline.subtitles import split_into_cues
from pipeline.transcriber import Segment


def _sentence(words, start=None, end=None, text=None):
    ws = [[w, s, e] for w, s, e in words]
    return Segment(id=0, start=start if start is not None else ws[0][1], end=end if end is not None else ws[-1][2],
                   text=text or " ".join(w for w, _, _ in words), words=ws)


def test_breaks_at_punctuation_within_limits():
    words = [(f"word{i}" + ("," if i == 5 else ""), i * 0.4, i * 0.4 + 0.3) for i in range(12)]  # 12 words ~ 5-6 chars each
    cues = split_into_cues([_sentence(words)], max_chars=40, max_duration=10.0, min_duration=0.0)
    assert len(cues) >= 2
    assert all(len(c.text) <= 40 for c in cues)
    assert cues[0].text.endswith("word5,")           # preferred break at the comma
    assert cues[0].start == 0.0 and cues[0].end == pytest.approx(5 * 0.4 + 0.3)
    assert cues[1].start == pytest.approx(6 * 0.4)
    assert [c.id for c in cues] == list(range(len(cues)))


def test_breaks_on_max_duration():
    words = [(f"w{i}", i * 2.0, i * 2.0 + 1.5) for i in range(6)]   # 12 s of speech, short words
    cues = split_into_cues([_sentence(words)], max_chars=84, max_duration=5.0, min_duration=0.0)
    assert len(cues) >= 3
    assert all(c.end - c.start <= 5.0 + 1e-6 for c in cues)


def test_short_cue_is_merged_into_previous():
    words = [("Hello", 0.0, 0.5), ("there,", 0.6, 1.0), ("ok.", 1.05, 1.2)]
    cues = split_into_cues([_sentence(words)], max_chars=16, max_duration=6.0, min_duration=1.0)
    # "Hello there," (12 chars) then "ok." would be a 0.15 s cue → merged back when it fits (16 chars combined)
    assert cues[-1].end == pytest.approx(1.2)
    assert sum(len(c.text.split()) for c in cues) == 3
    assert len(cues) == 1
    assert cues[0].text == "Hello there, ok."


def test_fallback_without_words_splits_by_chars_and_time():
    seg = Segment(id=0, start=10.0, end=20.0, text=" ".join(["abcde"] * 20))   # 119 chars, 10 s
    cues = split_into_cues([seg], max_chars=60, max_duration=60.0, min_duration=0.0)
    assert len(cues) == 2
    assert cues[0].start == 10.0 and cues[-1].end == 20.0
    assert cues[0].end == pytest.approx(cues[1].start)
    assert all(len(c.text) <= 60 for c in cues)


def test_sentence_within_limits_is_one_cue():
    words = [("Short", 0.0, 0.3), ("one.", 0.4, 0.8)]
    cues = split_into_cues([_sentence(words)], max_chars=84, max_duration=6.0, min_duration=1.0)
    assert len(cues) == 1 and cues[0].text == "Short one." and cues[0].words is None


def test_fallback_zero_duration_sentence_still_yields_positive_cues():
    text = " ".join("abcdefghijklmnopqrst")   # "a b c ... t", 20 single-letter words
    seg = Segment(id=0, start=5.0, end=5.0, text=text)
    cues = split_into_cues([seg], max_chars=10, max_duration=60.0, min_duration=0.0)
    assert len(cues) >= 2
    assert all(c.end > c.start for c in cues)
    for prev, nxt in zip(cues, cues[1:]):
        assert prev.end == pytest.approx(nxt.start)
    assert " ".join(c.text for c in cues) == text
