import threading, time
from dataclasses import replace
import numpy as np, soundfile as sf
from pipeline import fitter
from pipeline.transcriber import Segment

FCFG = dict(sec_per_syllable=0.2, overrun_tolerance=1.3, max_shorten_rounds=2, max_pause_borrow_s=0.6, margin_s=0.05, batch_chunk=32)


def _segs(n):
    return [Segment(id=i, start=i * 2.0, end=i * 2.0 + 1.5, text=f"src {i}") for i in range(n)]


class Probe:
    def __init__(self): self.lock = threading.Lock(); self.active = 0; self.max_active = 0; self.synth_calls = 0; self.overlap = False
    def translate(self, batch, budgets):
        with self.lock:
            self.active += 1; self.max_active = max(self.max_active, self.active)
        time.sleep(0.15)
        with self.lock: self.active -= 1
        return [replace(s, text=f"vi {s.id}", source_text=s.text) for s in batch]
    def synth_batch(self, texts, voice, paths):
        with self.lock:
            self.synth_calls += 1
            if self.active > 0: self.overlap = True
        time.sleep(0.1)
        for p in paths: sf.write(p, np.zeros(44100 // 2, dtype=np.float32), 44100)
        return list(paths)


def test_pipelined_overlaps_and_preserves_order(tmp_path):
    segs = _segs(8); slots = fitter.compute_slots(segs, 20.0, 0.6, 0.05)
    p = Probe()
    translated, units = fitter.run_pipelined(segs, slots, p.translate, lambda *a: "", None, p.synth_batch, str(tmp_path), FCFG, batch_size=2, workers=2)
    assert [s.id for s in translated] == list(range(8)) and [s.text for s in translated] == [f"vi {i}" for i in range(8)]
    assert [u.seg_id for u in units] == list(range(8)) and all(u.tts_dur > 0 for u in units)
    assert p.synth_calls == 4 and p.max_active >= 2 and p.overlap


def test_pipelined_matches_sequential(tmp_path):
    segs = _segs(5); slots = fitter.compute_slots(segs, 12.0, 0.6, 0.05)
    p = Probe()
    translated, units = fitter.run_pipelined(segs, slots, p.translate, lambda *a: "", None, p.synth_batch, str(tmp_path / "p"), FCFG, batch_size=2, workers=1)
    seq = p.translate(segs, [])
    u2 = fitter.build_units(seq, slots, {}, "nam-1"); fitter.fit_text(u2, lambda *a: "", FCFG)
    (tmp_path / "s").mkdir(); fitter.fit_audio(u2, None, str(tmp_path / "s"), FCFG, synth_batch=p.synth_batch)
    assert [(u.seg_id, u.text, round(u.tts_dur, 2)) for u in units] == [(u.seg_id, u.text, round(u.tts_dur, 2)) for u in u2]


def test_pipelined_propagates_translate_errors(tmp_path):
    segs = _segs(4); slots = fitter.compute_slots(segs, 10.0, 0.6, 0.05)
    def boom(batch, budgets): raise RuntimeError("llm down")
    import pytest
    with pytest.raises(RuntimeError, match="llm down"):
        fitter.run_pipelined(segs, slots, boom, lambda *a: "", None, Probe().synth_batch, str(tmp_path), FCFG, batch_size=2, workers=2)
