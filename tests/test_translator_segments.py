from unittest.mock import patch

from pipeline.transcriber import Segment
from pipeline.translator import translate_segments


def test_translate_segments_keeps_speaker_and_source_text():
    segs = [Segment(id=0, start=0.0, end=1.0, text="Hello there", speaker="SPEAKER_01")]
    with patch("pipeline.translator._translate_batch", return_value=["Xin chào"]):
        out = translate_segments(segs, "Vietnamese", client=object())
    assert out[0].text == "Xin chào"
    assert out[0].speaker == "SPEAKER_01"
    assert out[0].source_text == "Hello there"


def test_segment_source_text_defaults_empty():
    assert Segment(id=0, start=0.0, end=1.0, text="x").source_text == ""
