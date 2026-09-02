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

def test_absorption_respects_max_duration():
    segs = [Segment(id=i, start=i*0.6, end=i*0.6+0.4, text=f"Word{i}.", speaker="SPEAKER_00") for i in range(200)]
    out = merge_continuous_segments(segs, max_gap=1.0, max_duration=10.0, min_duration=2.5)
    for seg in out:
        assert seg.end - seg.start <= 10.0, f"Segment {seg.id} exceeds max_duration: {seg.end - seg.start}"
    assert len(out) > 1, "Expected multiple segments after respecting max_duration"


def test_short_segment_between_two_speakers_stays_alone():
    segs = [_s(0, 0.0, 3.0, "Hello.", "SPEAKER_00"), _s(1, 3.2, 4.0, "Hi.", "SPEAKER_01"), _s(2, 4.2, 7.0, "Okay.", "SPEAKER_00")]
    out = merge_continuous_segments(segs, min_duration=2.5)
    assert len(out) == 3, f"Expected 3 segments, got {len(out)}"
