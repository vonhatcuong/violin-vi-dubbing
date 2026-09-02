from pipeline import config as pipeline_config
from pipeline.transcriber import Segment, merge_continuous_segments


def setup_module(module):
    pipeline_config.load()


def _s(i, a, b, text, spk="SPEAKER_00"):
    return Segment(id=i, start=a, end=b, text=text, speaker=spk)


def test_short_segment_is_absorbed_into_previous_same_speaker():
    segs = [_s(0, 0.0, 3.0, "First sentence."), _s(1, 3.2, 4.0, "Yes."), _s(2, 4.5, 8.0, "Third one.")]
    out = merge_continuous_segments(segs, min_duration=2.5)
    assert [s.text for s in out] == ["First sentence. Yes.", "Third one."]
    assert out[0].start == 0.0 and out[0].end == 4.0


def test_short_segment_not_merged_across_speakers():
    segs = [_s(0, 0.0, 3.0, "Hello.", "SPEAKER_00"), _s(1, 3.2, 4.0, "Hi.", "SPEAKER_01")]
    out = merge_continuous_segments(segs, min_duration=2.5)
    assert len(out) == 2


def test_min_duration_zero_keeps_behaviour():
    segs = [_s(0, 0.0, 3.0, "Hello."), _s(1, 3.2, 4.0, "Hi.")]
    assert len(merge_continuous_segments(segs, min_duration=0.0)) == 2


def test_short_first_segment_absorbs_next():
    segs = [_s(0, 0.0, 0.8, "Okay."), _s(1, 1.0, 4.0, "So today we start.")]
    out = merge_continuous_segments(segs, min_duration=2.5)
    assert len(out) == 1 and out[0].text == "Okay. So today we start."
