"""Piecewise-linear map from source-video time to output-video time.

The merger may slow a unit's video (≤ 8 %) or trim it, so output timestamps
drift from the source. Units before/after the merger share ids and order,
and the gaps between units are copied 1:1, so the map is linear inside each
unit and a pure offset elsewhere. Used to re-time original-language ASR
sentences onto the dubbed video for subtitles.
"""

from __future__ import annotations

from typing import Callable

from .transcriber import Segment


def build_time_map(src: list[Segment], out: list[Segment]) -> Callable[[float], float]:
    knots: list[tuple[float, float]] = []
    for s, o in zip(src, out):
        knots.append((float(s.start), float(o.start)))
        knots.append((float(s.end), float(o.end)))

    def f(t: float) -> float:
        if not knots or t <= knots[0][0]:
            return float(t)
        for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
            if x0 <= t <= x1:
                if x1 - x0 < 1e-9:
                    return y1
                return y0 + (t - x0) * (y1 - y0) / (x1 - x0)
        x_last, y_last = knots[-1]
        return y_last + (t - x_last)

    return f
