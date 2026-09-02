"""Split ASR sentences into subtitle cues that follow the speech word by word.

Readability rules: ≤ max_chars per cue (≈ 2 lines × 42), ≤ max_duration
seconds, break preferably after , ; : . ! ?, never mid-word; cues shorter
than min_duration are merged into the previous cue when the text still fits.
Sentences without word timestamps fall back to character-proportional timing.
"""

from __future__ import annotations

import math

from .transcriber import Segment

_BREAK_PUNCT = (",", ";", ":", ".", "!", "?", "。", "，")


def _make_cue(words: list[list]) -> Segment:
    return Segment(id=0, start=float(words[0][1]), end=max(float(words[-1][2]), float(words[0][1]) + 0.05),
                   text=" ".join(w[0] for w in words).strip())


def _best_break(buf: list[list]) -> int:
    """Index to cut *buf* at (exclusive): last punctuation word past 40 % of the buffer, else the end."""
    floor = max(1, math.ceil(len(buf) * 0.4))
    for i in range(len(buf) - 1, floor - 1, -1):
        if buf[i][0].endswith(_BREAK_PUNCT):
            return i + 1
    return len(buf)


def _cues_from_words(seg: Segment, max_chars: int, max_duration: float) -> list[Segment]:
    cues: list[Segment] = []
    buf: list[list] = []
    for w in seg.words or []:
        if not str(w[0]).strip():
            continue
        cand = buf + [w]
        text_len = len(" ".join(x[0] for x in cand))
        dur = float(cand[-1][2]) - float(cand[0][1])
        if buf and (text_len > max_chars or dur > max_duration):
            k = _best_break(buf)
            cues.append(_make_cue(buf[:k]))
            buf = buf[k:] + [w]
        else:
            buf = cand
    if buf:
        cues.append(_make_cue(buf))
    return cues


def _cues_by_proportion(seg: Segment, max_chars: int) -> list[Segment]:
    words = seg.text.split()
    chunks: list[str] = []
    cur = ""
    for w in words:
        nxt = (cur + " " + w).strip()
        if cur and len(nxt) > max_chars:
            chunks.append(cur)
            cur = w
        else:
            cur = nxt
    if cur:
        chunks.append(cur)
    total = sum(len(c) for c in chunks) or 1
    dur = seg.end - seg.start
    out: list[Segment] = []
    t = seg.start
    for i, c in enumerate(chunks):
        end = seg.end if i == len(chunks) - 1 else t + dur * len(c) / total
        out.append(Segment(id=0, start=t, end=end, text=c))
        t = end
    return out


def split_into_cues(
    sentences: list[Segment], *, max_chars: int = 84, max_duration: float = 6.0, min_duration: float = 1.0,
) -> list[Segment]:
    cues: list[Segment] = []
    for seg in sentences:
        parts = _cues_from_words(seg, max_chars, max_duration) if seg.words else _cues_by_proportion(seg, max_chars)
        for p in parts:
            prev = cues[-1] if cues else None
            if (prev is not None and (p.end - p.start) < min_duration
                    and len(prev.text) + 1 + len(p.text) <= max_chars
                    and (p.end - prev.start) <= max_duration):
                cues[-1] = Segment(id=0, start=prev.start, end=p.end, text=(prev.text + " " + p.text).strip(),
                                   speaker=prev.speaker)
            else:
                cues.append(Segment(id=0, start=p.start, end=p.end, text=p.text, speaker=seg.speaker))
    for i, c in enumerate(cues):
        c.id = i
    return cues
