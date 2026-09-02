import pytest

from pipeline.timemap import build_time_map
from pipeline.transcriber import Segment


def _s(i, a, b):
    return Segment(id=i, start=a, end=b, text="x")


def test_identity_when_nothing_changed():
    f = build_time_map([_s(0, 1.0, 3.0), _s(1, 4.0, 6.0)], [_s(0, 1.0, 3.0), _s(1, 4.0, 6.0)])
    for t in (0.0, 1.0, 2.5, 3.5, 6.0, 9.0):
        assert f(t) == pytest.approx(t)


def test_stretched_unit_maps_linearly_and_shifts_the_rest():
    src = [_s(0, 2.0, 4.0), _s(1, 5.0, 6.0)]
    out = [_s(0, 2.0, 5.0), _s(1, 6.0, 7.0)]   # unit 0 stretched ×1.5, gap 4→5 copied 1:1 → 5→6
    f = build_time_map(src, out)
    assert f(1.0) == pytest.approx(1.0)      # before any unit: unchanged
    assert f(3.0) == pytest.approx(3.5)      # inside unit 0: linear
    assert f(4.0) == pytest.approx(5.0)
    assert f(4.5) == pytest.approx(5.5)      # in the gap: offset +1
    assert f(5.5) == pytest.approx(6.5)      # inside unit 1 (not stretched)
    assert f(8.0) == pytest.approx(9.0)      # after the last unit: offset +1


def test_empty_units_is_identity():
    f = build_time_map([], [])
    assert f(12.3) == pytest.approx(12.3)
