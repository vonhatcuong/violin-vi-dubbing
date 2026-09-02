import numpy as np
import soundfile as sf

from pipeline import voices
from pipeline.transcriber import Segment


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
