import unicodedata
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from pipeline import config as pipeline_config
from pipeline import tts, tts_vieneu


class FakeEngine:
    """Mimics vieneu.Vieneu: records calls; returns `seconds` of 48 kHz sine per infer."""

    sample_rate = 48000

    def __init__(self, seconds=1.0):
        self.seconds = seconds
        self.calls: list[dict] = []
        self.batch_calls: list[dict] = []
        self.enrolled: list[tuple[str, str]] = []

    def _sine(self):
        n = int(self.sample_rate * self.seconds)
        return (0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / self.sample_rate)).astype(np.float32)

    def infer(self, text, ref_audio=None, voice=None, **kwargs):
        self.calls.append(dict(text=text, ref_audio=ref_audio, voice=voice, **kwargs))
        return self._sine()

    def infer_batch(self, texts, voice=None, **kwargs):
        self.batch_calls.append(dict(texts=list(texts), voice=voice, **kwargs))
        return [self._sine() for _ in texts]

    def add_voice(self, name, ref_audio, *, denoise=True):
        self.enrolled.append((name, str(ref_audio)))

    def list_preset_voices(self):
        return [("Phạm Tuyên — Bắc, Natural", "Phạm Tuyên"), ("Ngọc Huyền — Bắc, Natural", "Ngọc Huyền")]


@pytest.fixture
def env(tmp_path, monkeypatch):
    sr = 24000
    sf.write(tmp_path / "nam-1.wav", np.zeros(sr * 5, dtype=np.float32), sr)
    (tmp_path / "catalog.yaml").write_text(yaml.safe_dump({"voices": [
        {"name": "nam-1", "gender": "male", "wav": "nam-1.wav", "ref_text": "xin chào các bạn"}]}), encoding="utf-8")
    cfg = pipeline_config.load()
    monkeypatch.setitem(cfg["voices"], "bank", str(tmp_path))
    monkeypatch.setitem(cfg["tts"], "sentence_tail_silence_ms", 250)
    monkeypatch.setitem(cfg["tts"], "tail_silence_ms", 120)
    monkeypatch.setitem(cfg["vieneu"], "loanwords", {})
    monkeypatch.setitem(cfg["vieneu"], "normalize_numbers", True)
    monkeypatch.setattr(tts_vieneu, "_ENROLLED", set())
    return tmp_path


def test_synthesize_segment_writes_44k_mono_with_sentence_tail(env, tmp_path):
    engine = FakeEngine(seconds=1.0)
    out = tts_vieneu.synthesize_segment("Xin chào.", "Phạm Tuyên", str(tmp_path / "o.wav"), engine, language="vi")
    info = sf.info(out)
    assert info.samplerate == 44100 and info.channels == 1
    assert abs(info.duration - 1.25) < 0.03


def test_preset_voice_is_passed_by_name_without_enrollment(env, tmp_path):
    engine = FakeEngine()
    tts_vieneu.synthesize_segment("Một câu", "Phạm Tuyên", str(tmp_path / "o.wav"), engine, language="vi")
    assert engine.calls[0]["voice"] == "Phạm Tuyên" and engine.calls[0]["ref_audio"] is None
    assert engine.enrolled == []
    assert engine.calls[0]["temperature"] == pytest.approx(0.8)


def test_bank_voice_is_enrolled_once_then_used_by_name(env, tmp_path):
    engine = FakeEngine()
    tts_vieneu.synthesize_segment("Câu một", "nam-1", str(tmp_path / "a.wav"), engine, language="vi")
    tts_vieneu.synthesize_segment("Câu hai", "nam-1", str(tmp_path / "b.wav"), engine, language="vi")
    assert len(engine.enrolled) == 1 and engine.enrolled[0][0] == "nam-1"
    assert Path(engine.enrolled[0][1]).name == "nam-1.wav"
    assert [c["voice"] for c in engine.calls] == ["nam-1", "nam-1"]


def test_numbers_are_spelled_out_and_case_is_kept(env, tmp_path):
    engine = FakeEngine()
    tts_vieneu.synthesize_segment("Có 23 GPU mới.", "Phạm Tuyên", str(tmp_path / "o.wav"), engine, language="vi")
    assert engine.calls[0]["text"] == "Có hai mươi ba GPU mới."


def test_unknown_voice_raises_helpful_error(env, tmp_path):
    engine = FakeEngine()
    with pytest.raises(KeyError, match="Phạm Tuyên"):
        tts_vieneu.synthesize_segment("x", "Giọng Không Có", str(tmp_path / "o.wav"), engine, language="vi")


def test_native_voices_come_from_config_defaults(env):
    assert tts_vieneu.native_voices_for("vi") == ["Thanh Bình", "Ngọc Huyền"]


def test_make_synthesizer_uses_vieneu_backend(env, tmp_path, monkeypatch):
    cfg = pipeline_config.get()
    monkeypatch.setitem(cfg["models"], "tts", {"provider": "vieneu", "model": "vieneu-v3-turbo"})
    engine = FakeEngine(seconds=0.5)
    monkeypatch.setattr(tts_vieneu, "get_shared_tts", lambda: engine)
    synth = tts.make_synthesizer(language="vi")
    out = synth("Xin chào", "Phạm Tuyên", str(tmp_path / "s.wav"), 1.0)
    assert Path(out).exists() and len(engine.calls) == 1


def test_nfc_normalization_always_applied_even_when_normalize_numbers_is_off(env, tmp_path, monkeypatch):
    cfg = pipeline_config.get()
    monkeypatch.setitem(cfg["vieneu"], "normalize_numbers", False)
    engine = FakeEngine()
    decomposed = "Cha" + "̀" + "o bạn"  # NFD "Chào bạn"
    tts_vieneu.synthesize_segment(decomposed, "Phạm Tuyên", str(tmp_path / "o.wav"), engine, language="vi")
    assert engine.calls[0]["text"] == "Chào bạn"
    tts_vieneu.synthesize_segment("Có 23 GPU", "Phạm Tuyên", str(tmp_path / "p.wav"), engine, language="vi")
    assert engine.calls[1]["text"] == "Có 23 GPU"


def test_voice_name_given_as_nfd_resolves_to_preset(env, tmp_path):
    engine = FakeEngine()
    decomposed_voice = unicodedata.normalize("NFD", "Phạm Tuyên")
    tts_vieneu.synthesize_segment("x", decomposed_voice, str(tmp_path / "o.wav"), engine, language="vi")
    assert engine.calls[0]["voice"] == "Phạm Tuyên"


def test_synthesize_batch_writes_all_files_with_one_engine_call(env, tmp_path):
    engine = FakeEngine(seconds=1.0)
    paths = [str(tmp_path / f"b{i}.wav") for i in range(3)]
    out = tts_vieneu.synthesize_batch(["Câu một.", "Câu hai", "Câu ba."], "Phạm Tuyên", paths, engine, language="vi")
    assert out == paths and len(engine.batch_calls) == 1 and engine.calls == []
    assert engine.batch_calls[0]["voice"] == "Phạm Tuyên" and engine.batch_calls[0]["texts"][1] == "Câu hai"
    durs = [sf.info(p).duration for p in paths]
    assert durs[0] == pytest.approx(1.25, abs=0.03) and durs[1] == pytest.approx(1.12, abs=0.03)   # sentence tail 250 ms vs 120 ms
    assert all(sf.info(p).samplerate == 44100 and sf.info(p).channels == 1 for p in paths)


def test_synthesize_batch_raises_when_engine_returns_fewer_clips_than_requested(env, tmp_path):
    engine = FakeEngine(seconds=1.0)
    engine.infer_batch = lambda texts, voice=None, **kwargs: [engine._sine() for _ in texts[:-1]]  # one short
    paths = [str(tmp_path / f"c{i}.wav") for i in range(3)]
    with pytest.raises(RuntimeError, match="2 clips for 3 texts"):
        tts_vieneu.synthesize_batch(["Một.", "Hai.", "Ba."], "Phạm Tuyên", paths, engine, language="vi")
    assert not Path(paths[-1]).exists()


def test_make_batch_synthesizer_none_when_disabled(env, monkeypatch):
    cfg = pipeline_config.get()
    monkeypatch.setitem(cfg["models"], "tts", {"provider": "vieneu", "model": "vieneu-v3-turbo"})
    monkeypatch.setattr(tts_vieneu, "get_shared_tts", lambda: FakeEngine())
    assert callable(tts.make_batch_synthesizer(language="vi"))
    monkeypatch.setitem(cfg["vieneu"], "batch_tts", False)
    assert tts.make_batch_synthesizer(language="vi") is None
