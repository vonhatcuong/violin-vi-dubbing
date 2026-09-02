from pathlib import Path

import pytest
import yaml

from pipeline import voices


@pytest.fixture
def bank(tmp_path: Path) -> Path:
    (tmp_path / "nam-1.wav").write_bytes(b"RIFF")
    (tmp_path / "nu-1.wav").write_bytes(b"RIFF")
    (tmp_path / "catalog.yaml").write_text(yaml.safe_dump({"voices": [
        {"name": "nam-1", "gender": "male", "wav": "nam-1.wav", "ref_text": "xin chào", "description": "nam bắc"},
        {"name": "nu-1", "gender": "female", "wav": "nu-1.wav", "ref_text": "xin chào", "description": "nữ bắc"},
    ]}), encoding="utf-8")
    return tmp_path


def test_load_catalog_resolves_paths(bank):
    cat = voices.load_catalog(bank)
    assert set(cat) == {"nam-1", "nu-1"}
    assert cat["nam-1"].ref_wav == bank / "nam-1.wav"
    assert cat["nu-1"].gender == "female"


def test_native_voices_first_male_then_female(bank):
    assert voices.native_voices_for("vi", bank=bank) == ["nam-1", "nu-1"]


def test_missing_bank_raises_helpful_error(tmp_path):
    with pytest.raises(RuntimeError, match="make_ref_clip"):
        voices.native_voices_for("vi", bank=tmp_path)


def test_assign_voices_uses_map_then_gender_then_default(bank):
    out = voices.assign_voices(
        ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
        default_voice="nam-1",
        voice_map={"SPEAKER_00": "nu-1"},
        genders={"SPEAKER_01": "female"},
        bank=bank,
    )
    assert out == {"SPEAKER_00": "nu-1", "SPEAKER_01": "nu-1", "SPEAKER_02": "nam-1"}


def test_get_voice_unknown_name(bank):
    with pytest.raises(KeyError):
        voices.get_voice("nope", bank=bank)
