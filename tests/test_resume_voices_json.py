from pathlib import Path

from resume_from_segments import _voices_json_path


def test_voices_json_path_strips_stage_suffix():
    assert _voices_json_path("/tmp/out.transcribed.segments.json", "transcribed") == Path("/tmp/out.voices.json")
    assert _voices_json_path("/tmp/out.diarized.segments.json", "diarized") == Path("/tmp/out.voices.json")


def test_voices_json_path_none_when_suffix_does_not_match():
    assert _voices_json_path("/tmp/weird_name.json", "transcribed") is None
