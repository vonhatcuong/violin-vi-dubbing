from pathlib import Path

from resume_from_segments import _load_voice_map, _voices_json_path


def test_voices_json_path_strips_stage_suffix():
    assert _voices_json_path("/tmp/out.transcribed.segments.json", "transcribed") == Path("/tmp/out.voices.json")
    assert _voices_json_path("/tmp/out.diarized.segments.json", "diarized") == Path("/tmp/out.voices.json")


def test_voices_json_path_none_when_suffix_does_not_match():
    assert _voices_json_path("/tmp/weird_name.json", "transcribed") is None


def test_load_voice_map_missing_file_returns_none(tmp_path):
    assert _load_voice_map(tmp_path / "nope.voices.json") is None


def test_load_voice_map_null_content_returns_none_without_raising(tmp_path):
    p = tmp_path / "out.voices.json"
    p.write_text("null", encoding="utf-8")
    assert _load_voice_map(p) is None


def test_load_voice_map_empty_file_returns_none_without_raising(tmp_path):
    p = tmp_path / "out.voices.json"
    p.write_text("", encoding="utf-8")
    assert _load_voice_map(p) is None


def test_load_voice_map_list_content_returns_none_without_raising(tmp_path):
    p = tmp_path / "out.voices.json"
    p.write_text("[]", encoding="utf-8")
    assert _load_voice_map(p) is None


def test_load_voice_map_drops_bad_entries_keeps_good_ones(tmp_path):
    p = tmp_path / "out.voices.json"
    p.write_text(
        '{"SPEAKER_00": "Thanh Binh", "SPEAKER_01": 5, "": "X", "SPEAKER_02": ""}',
        encoding="utf-8",
    )
    assert _load_voice_map(p) == {"SPEAKER_00": "Thanh Binh"}


def test_load_voice_map_valid_dict_passes_through(tmp_path):
    p = tmp_path / "out.voices.json"
    p.write_text('{"SPEAKER_00": "Thanh Binh", "SPEAKER_01": "Ngoc Huyen"}', encoding="utf-8")
    assert _load_voice_map(p) == {"SPEAKER_00": "Thanh Binh", "SPEAKER_01": "Ngoc Huyen"}
