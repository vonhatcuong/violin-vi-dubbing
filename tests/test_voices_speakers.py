from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from pipeline import voices
from pipeline.transcriber import Segment

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_assign_voices_round_robins_speaker_voices_by_first_appearance():
    out = voices.assign_voices(
        ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"],
        default_voice="fallback",
        speaker_voices=["Thanh Bình", "Ngọc Huyền", "Minh Đức"],
    )
    assert out == {
        "SPEAKER_00": "Thanh Bình",
        "SPEAKER_01": "Ngọc Huyền",
        "SPEAKER_02": "Minh Đức",
        "SPEAKER_03": "Thanh Bình",  # wraps around
    }


def test_assign_voices_falls_back_to_default_when_no_speaker_voices():
    out = voices.assign_voices(["SPEAKER_00", "SPEAKER_01"], default_voice="fallback")
    assert out == {"SPEAKER_00": "fallback", "SPEAKER_01": "fallback"}


def test_assign_voices_matches_gender_from_speaker_voices():
    out = voices.assign_voices(
        ["A", "B"],
        default_voice="fallback",
        genders={"A": "male", "B": "female"},
        speaker_voices=["Thanh Bình", "Ngọc Huyền", "Minh Đức", "Trúc Ly", "Thái Sơn", "Mai Anh"],
    )
    assert out == {"A": "Thanh Bình", "B": "Ngọc Huyền"}


def test_assign_voices_voice_map_wins_over_gender_and_round_robin():
    out = voices.assign_voices(
        ["A", "B"],
        default_voice="fallback",
        voice_map={"A": "Custom Voice"},
        genders={"A": "female"},
        speaker_voices=["Thanh Bình", "Ngọc Huyền"],
    )
    assert out["A"] == "Custom Voice"
    assert out["B"] == "Thanh Bình"


def test_assign_voices_uses_custom_preset_genders_map():
    out = voices.assign_voices(
        ["A"],
        default_voice="fallback",
        genders={"A": "female"},
        speaker_voices=["OnlyVoice"],
        preset_genders={"OnlyVoice": "female"},
    )
    assert out == {"A": "OnlyVoice"}


def test_assign_voices_cycles_through_gender_pool_instead_of_collapsing():
    # Three male speakers must NOT all land on the first male match — each
    # should get a different preset from the male subset, in order.
    out = voices.assign_voices(
        ["A", "B", "C"],
        default_voice="fallback",
        genders={"A": "male", "B": "male", "C": "male"},
        speaker_voices=["Thanh Bình", "Ngọc Huyền", "Minh Đức", "Trúc Ly", "Thái Sơn", "Mai Anh"],
    )
    assert out == {"A": "Thanh Bình", "B": "Minh Đức", "C": "Thái Sơn"}


def test_assign_voices_gender_cursor_wraps_around():
    out = voices.assign_voices(
        ["A", "B", "C", "D"],
        default_voice="fallback",
        genders={"A": "female", "B": "female", "C": "female", "D": "female"},
        speaker_voices=["Thanh Bình", "Ngọc Huyền", "Minh Đức", "Trúc Ly", "Thái Sơn", "Mai Anh"],
    )
    assert out == {"A": "Ngọc Huyền", "B": "Trúc Ly", "C": "Mai Anh", "D": "Ngọc Huyền"}


def test_assign_voices_seed_voice_goes_to_first_speaker_not_in_voice_map():
    out = voices.assign_voices(
        ["A", "B", "C"],
        default_voice="fallback",
        genders={"B": "male", "C": "female"},
        speaker_voices=["Thanh Bình", "Ngọc Huyền", "Minh Đức", "Trúc Ly", "Thái Sơn", "Mai Anh"],
        seed_voice="Custom Seed",
    )
    assert out["A"] == "Custom Seed"
    assert out["B"] == "Thanh Bình"
    assert out["C"] == "Ngọc Huyền"


def test_assign_voices_seed_voice_excluded_from_gender_cursor_when_possible():
    out = voices.assign_voices(
        ["A", "B"],
        default_voice="fallback",
        genders={"B": "male"},
        speaker_voices=["Thanh Bình", "Minh Đức", "Thái Sơn"],
        seed_voice="Thanh Bình",
    )
    assert out["A"] == "Thanh Bình"
    assert out["B"] == "Minh Đức"  # cursor skips the seed voice since another option exists


def test_assign_voices_seed_voice_yields_to_explicit_voice_map_speaker():
    out = voices.assign_voices(
        ["A", "B"],
        default_voice="fallback",
        voice_map={"A": "Mapped Voice"},
        speaker_voices=["Thanh Bình", "Ngọc Huyền"],
        seed_voice="Custom Seed",
    )
    assert out["A"] == "Mapped Voice"
    assert out["B"] == "Custom Seed"  # first speaker NOT covered by voice_map


def test_preset_genders_matches_config_default():
    cfg = yaml.safe_load((_REPO_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    assert voices.PRESET_GENDERS == cfg["voices"]["preset_genders"]


def _sawtooth(freq: float, duration_s: float, sr: int, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    phase = (t * freq) % 1.0
    return ((2 * phase - 1) * amp).astype(np.float32)


def test_guess_genders_from_f0(tmp_path):
    sr = 16000
    male_wave = _sawtooth(120.0, 3.0, sr)     # low F0 → male
    female_wave = _sawtooth(220.0, 3.0, sr)   # higher F0 → female
    silence = np.zeros(int(1.0 * sr), dtype=np.float32)
    audio = np.concatenate([male_wave, female_wave, silence])

    wav_path = tmp_path / "speakers.wav"
    sf.write(wav_path, audio, sr)

    segments = [
        Segment(id=0, start=0.0, end=3.0, text="a", speaker="SPEAKER_00"),
        Segment(id=1, start=3.0, end=6.0, text="b", speaker="SPEAKER_01"),
        Segment(id=2, start=6.0, end=7.0, text="c", speaker="SPEAKER_02"),  # silence only
    ]

    genders = voices.guess_genders(str(wav_path), segments)

    assert genders["SPEAKER_00"] == "male"
    assert genders["SPEAKER_01"] == "female"
    assert "SPEAKER_02" not in genders
