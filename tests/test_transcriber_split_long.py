from pipeline import config as pipeline_config
from pipeline.transcriber import Segment, merge_continuous_segments, split_long_segments


def _sent(words, speaker="SPEAKER_00"):
    ws = [[w, s, e] for w, s, e in words]
    return Segment(id=0, start=ws[0][1], end=ws[-1][2], text=" ".join(w for w, _, _ in words), speaker=speaker, words=ws)


def test_splits_at_comma_nearest_middle():
    words = [(f"w{i}" + ("," if i == 9 else ""), i * 1.0, i * 1.0 + 0.8) for i in range(20)]   # 0–19.8 s, comma after w9
    out = split_long_segments([_sent(words)], max_seconds=12.0)
    assert [s.text.split()[0] for s in out] == ["w0", "w10"]
    assert out[0].end == 9.8 and out[1].start == 10.0
    assert out[0].words[-1][0] == "w9," and [s.id for s in out] == [0, 1]


def test_splits_at_largest_gap_without_punctuation():
    words = [(f"w{i}", i * 1.0 + (2.0 if i >= 12 else 0.0), i * 1.0 + 0.8 + (2.0 if i >= 12 else 0.0)) for i in range(20)]
    out = split_long_segments([_sent(words)], max_seconds=12.0)
    assert len(out) == 2 and out[0].words[-1][0] == "w11" and out[1].words[0][0] == "w12"


def test_short_or_wordless_segments_untouched():
    short = _sent([("a", 0.0, 0.5), ("b.", 0.6, 1.0)])
    nowords = Segment(id=0, start=0.0, end=30.0, text="x " * 50)
    out = split_long_segments([short, nowords], max_seconds=12.0)
    assert out[0].text == "a b." and out[1].end == 30.0 and len(out) == 2


def test_min_piece_respected_and_recursive():
    words = [(f"w{i}" + ("," if i in (1, 15) else ""), i * 1.0, i * 1.0 + 0.8) for i in range(30)]   # 30 s; comma after w1 (too early) and w15
    out = split_long_segments([_sent(words)], max_seconds=12.0, min_piece_seconds=2.5)
    assert all((s.end - s.start) <= 12.0 + 1e-6 for s in out) and all((s.end - s.start) >= 2.5 for s in out)
    assert sum(len(s.words) for s in out) == 30


def test_merge_keeps_words_when_both_have_them():
    pipeline_config.load()
    a = _sent([("Hello", 0.0, 0.4), ("there", 0.5, 0.9)])
    b = _sent([("friend.", 1.0, 1.4)])
    out = merge_continuous_segments([a, b], min_duration=0.0)
    assert len(out) == 1 and out[0].words == [["Hello", 0.0, 0.4], ["there", 0.5, 0.9], ["friend.", 1.0, 1.4]]
