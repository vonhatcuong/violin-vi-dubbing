from types import SimpleNamespace

from pipeline.transcriber import _split_words_into_sentences


def _w(word, start, end):
    return SimpleNamespace(word=word, start=start, end=end)


def test_sentences_carry_word_timestamps_with_offset():
    words = [_w(" Hello", 0.0, 0.4), _w(" there.", 0.5, 0.9), _w(" Second", 1.2, 1.6), _w(" one.", 1.7, 2.0)]
    out = _split_words_into_sentences(words, offset=10.0)
    assert [s.text for s in out] == ["Hello there.", "Second one."]
    assert out[0].words == [["Hello", 10.0, 10.4], ["there.", 10.5, 10.9]]
    assert out[1].start == 11.2 and out[1].words[-1] == ["one.", 11.7, 12.0]
