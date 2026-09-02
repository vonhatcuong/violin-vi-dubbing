import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline import fitter
from pipeline.transcriber import Segment

FCFG = dict(sec_per_syllable=0.2, overrun_tolerance=1.3, max_shorten_rounds=2,
            max_pause_borrow_s=0.6, margin_s=0.05)


def _segs():
    return [
        Segment(id=0, start=0.0, end=2.0, text="một hai ba bốn năm", source_text="one two"),
        Segment(id=1, start=3.0, end=5.0, text="sáu bảy", source_text="six seven"),
        Segment(id=2, start=5.2, end=7.0, text="tám", source_text="eight"),
    ]


def test_compute_slots_borrows_pause_up_to_next_onset():
    slots = fitter.compute_slots(_segs(), total_duration=10.0, max_borrow_s=0.6, margin_s=0.05)
    assert slots == pytest.approx([2.6, 5.15, 7.6])


def test_budgets_for_uses_slot_and_syllable_rate():
    segs = _segs()
    slots = fitter.compute_slots(segs, 10.0, 0.6, 0.05)
    budgets = fitter.budgets_for(segs, slots, sec_per_syllable=0.2)
    assert budgets[0] == (pytest.approx(2.6), 13)


def test_fit_text_shortens_until_within_tolerance():
    segs = _segs()
    slots = [1.0, 5.15, 7.6]                      # unit 0 gets a 1.0 s budget (5 syll ok, tol → 6.5)
    units = fitter.build_units(segs, slots, {}, "nam-1")
    units[0].text = "một hai ba bốn năm sáu bảy tám chín mười"   # 10 syll → est 2.0 s > 1.3
    calls = []

    def shorten(src, cur, syl, sec):
        calls.append((src, cur, syl, sec))
        return "một hai ba bốn"                    # 4 syll → est 0.8 s

    fitter.fit_text(units, shorten, FCFG)
    assert calls == [("one two", "một hai ba bốn năm sáu bảy tám chín mười", 5, pytest.approx(1.0))]
    assert units[0].text == "một hai ba bốn" and units[0].rounds == 1 and units[0].strategy == "shortened"
    assert units[1].rounds == 0


def test_fit_text_stops_when_no_progress():
    units = fitter.build_units(_segs(), [1.0, 5.15, 7.6], {}, "nam-1")
    units[0].text = "a b c d e f g h i j"
    fitter.fit_text(units, lambda *a: "a b c d e f g h i j", FCFG)
    assert units[0].rounds == 1


def _fake_synth(tmp_path):
    calls = []

    def synth(text, voice, out_path, speed=1.0):
        calls.append((text, voice, speed))
        dur = 0.2 * len(text.split()) / speed
        sr = 44100
        sf.write(out_path, np.zeros(int(sr * dur), dtype=np.float32), sr)
        return out_path

    return synth, calls


def test_fit_audio_measures_and_flags_overrun(tmp_path):
    units = fitter.build_units(_segs(), [1.0, 5.15, 7.6], {}, "nam-1")   # unit0 budget 1.0 s
    units[0].text = "a b c d e f g h"                                    # 8 syll = 1.6 s > 1.0
    synth, calls = _fake_synth(tmp_path)
    fitter.fit_audio(units, synth, str(tmp_path), FCFG)
    u = units[0]
    assert u.tts_dur == pytest.approx(1.6, abs=0.02)
    assert u.strategy == "over" and u.over_s == pytest.approx(0.6, abs=0.02)
    assert [c for c in calls if c[0] == "a b c d e f g h"] == [("a b c d e f g h", "nam-1", 1.0)]   # synthesized once
    assert units[1].strategy == "natural" and units[1].over_s == 0.0
    assert all(Path(x.tts_path).exists() for x in units)


def test_fit_audio_keeps_shortened_flag_when_still_over(tmp_path):
    units = fitter.build_units(_segs(), [1.0, 5.15, 7.6], {}, "nam-1")
    units[0].text = "a b c d e f g h"
    units[0].rounds, units[0].strategy = 1, "shortened"
    synth, _ = _fake_synth(tmp_path)
    fitter.fit_audio(units, synth, str(tmp_path), FCFG)
    assert units[0].strategy == "shortened+over"


def test_apply_units_extends_end_to_borrow_pause(tmp_path):
    segs = _segs()
    units = fitter.build_units(segs, [2.6, 5.15, 7.6], {}, "nam-1")
    units[0].tts_dur, units[0].tts_path = 2.4, "a.wav"     # longer than 2.0 → end becomes 2.4
    units[1].tts_dur, units[1].tts_path = 1.0, "b.wav"     # shorter → keep 5.0
    units[2].tts_dur, units[2].tts_path = 9.0, "c.wav"     # way over → capped at slot_end 7.6
    out, paths = fitter.apply_units(units, segs)
    assert [round(s.end, 2) for s in out] == [2.4, 5.0, 7.6]
    assert paths == ["a.wav", "b.wav", "c.wav"]
    assert out[0].source_text == "one two"


def test_save_units_writes_json(tmp_path):
    units = fitter.build_units(_segs(), [2.6, 5.15, 7.6], {}, "nam-1")
    p = tmp_path / "x.fit.units.json"
    fitter.save_units(units, p)
    data = json.loads(p.read_text())
    assert data["count"] == 3 and data["units"][0]["voice"] == "nam-1"


def _fake_batch_synth(tmp_path):
    calls = []

    def synth_batch(texts, voice, paths):
        calls.append((list(texts), voice, list(paths)))
        for text, path in zip(texts, paths):
            sf.write(path, np.zeros(int(44100 * 0.2 * len(text.split())), dtype=np.float32), 44100)
        return list(paths)

    return synth_batch, calls


def test_fit_audio_uses_batch_synth_grouped_by_voice(tmp_path):
    units = fitter.build_units(_segs(), [1.0, 5.15, 7.6], {"SPEAKER_00": "nam-1"}, "nam-1")
    units[1].voice = "nu-1"
    units[0].text = "a b c d e f g h"
    synth, calls = _fake_synth(tmp_path)
    synth_batch, bcalls = _fake_batch_synth(tmp_path)
    fitter.fit_audio(units, synth, str(tmp_path), FCFG, synth_batch=synth_batch)
    assert calls == []                                   # per-unit synth not used
    assert [(c[1], len(c[0])) for c in bcalls] == [("nam-1", 2), ("nu-1", 1)]
    assert units[0].strategy == "over" and units[0].over_s == pytest.approx(0.6, abs=0.02)
    assert all(Path(u.tts_path).exists() for u in units)


def test_fit_audio_batch_and_serial_agree(tmp_path):
    a = fitter.build_units(_segs(), [2.6, 5.15, 7.6], {}, "nam-1")
    b = fitter.build_units(_segs(), [2.6, 5.15, 7.6], {}, "nam-1")
    synth, _ = _fake_synth(tmp_path / "s")
    synth_batch, _ = _fake_batch_synth(tmp_path / "b")
    (tmp_path / "s").mkdir(); (tmp_path / "b").mkdir()
    fitter.fit_audio(a, synth, str(tmp_path / "s"), FCFG)
    fitter.fit_audio(b, synth, str(tmp_path / "b"), FCFG, synth_batch=synth_batch)
    assert [(round(u.tts_dur, 3), u.over_s, u.strategy) for u in a] == \
        [(round(u.tts_dur, 3), u.over_s, u.strategy) for u in b]
