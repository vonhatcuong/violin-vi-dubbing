import numpy as np, pytest, soundfile as sf
from pipeline import fitter
from pipeline.transcriber import Segment

FCFG = dict(sec_per_syllable=0.2, overrun_tolerance=1.3, max_shorten_rounds=2, max_pause_borrow_s=0.6, margin_s=0.05, min_fill=0.6, min_tempo=0.85)


def _synth_fixed(seconds):
    def synth(text, voice, out_path, speed=1.0):
        sf.write(out_path, np.zeros(int(44100 * seconds), dtype=np.float32), 44100); return out_path
    return synth


def test_underfilled_unit_is_slowed_to_min_tempo(tmp_path):
    seg = Segment(id=0, start=0.0, end=10.0, text="ngắn")
    units = fitter.build_units([seg], [10.0], {}, "nam-1")
    fitter.fit_audio(units, _synth_fixed(3.0), str(tmp_path), FCFG)
    u = units[0]
    assert u.tempo == pytest.approx(0.85) and u.strategy == "slowed"
    assert u.tts_dur == pytest.approx(3.0 / 0.85, abs=0.05)


def test_partially_underfilled_unit_gets_intermediate_tempo(tmp_path):
    seg = Segment(id=0, start=0.0, end=10.0, text="x")
    units = fitter.build_units([seg], [10.0], {}, "nam-1")
    fitter.fit_audio(units, _synth_fixed(5.4), str(tmp_path), FCFG)    # 54 % filled → tempo 0.9
    assert units[0].tempo == pytest.approx(0.9, abs=0.01) and units[0].tts_dur == pytest.approx(6.0, abs=0.05)


def test_well_filled_unit_untouched(tmp_path):
    seg = Segment(id=0, start=0.0, end=10.0, text="x")
    units = fitter.build_units([seg], [10.0], {}, "nam-1")
    fitter.fit_audio(units, _synth_fixed(8.0), str(tmp_path), FCFG)
    assert units[0].tempo == 1.0 and units[0].strategy == "natural"


def test_min_fill_zero_disables(tmp_path):
    seg = Segment(id=0, start=0.0, end=10.0, text="x")
    units = fitter.build_units([seg], [10.0], {}, "nam-1")
    fitter.fit_audio(units, _synth_fixed(3.0), str(tmp_path), FCFG | {"min_fill": 0.0})
    assert units[0].tempo == 1.0 and units[0].tts_dur == pytest.approx(3.0, abs=0.02)


def test_min_fill_above_one_clamped_to_one(tmp_path, capsys):
    seg = Segment(id=0, start=0.0, end=10.0, text="x")
    units = fitter.build_units([seg], [10.0], {}, "nam-1")
    fitter.fit_audio(units, _synth_fixed(9.0), str(tmp_path), FCFG | {"min_fill": 6})
    # min_fill=6 must behave exactly like min_fill=1.0 (tempo = max(min_tempo, 9/10)).
    assert units[0].tempo == pytest.approx(0.9, abs=0.01)
    assert "clamp" in capsys.readouterr().out.lower()
