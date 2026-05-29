"""Fetch & normalize source captions (YouTube etc.) into Segment[].

When a video URL already ships captions we prefer them over re-running Whisper:
faster, cheaper, and often more accurate for proper nouns.

- Manual captions are used as-is (they already carry punctuation).
- Automatic captions are word-level but unpunctuated. Rather than a slow LLM
  punctuation-restore pass, we cut them into short segments at speech pauses
  (a natural clause boundary) with word-level timestamps. The downstream
  translation step then produces properly punctuated target-language text, so
  the dub and subtitles read correctly without an extra model call.

Any failure returns None so the caller falls back to Whisper.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass

from .languages import language_code
from .transcriber import Segment

_MAX_WORD_DUR = 0.6  # cap on a word's spoken length, so silent gaps survive as pauses


@dataclass
class _Word:
    text: str
    start: float
    end: float


@dataclass
class _TrackRef:
    kind: str   # "manual" | "auto"
    lang: str
    url: str


def _source_lang_code(info: dict, source_language: str) -> str | None:
    if source_language and source_language.lower() != "auto-detect":
        return language_code(source_language)
    lang = info.get("language")
    return lang.split("-")[0] if lang else None


def _json3_url(track_list) -> str | None:
    for e in track_list or []:
        if e.get("ext") == "json3":
            return e.get("url")
    return None


def _select_track(info: dict, src: str | None) -> _TrackRef | None:
    subs = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}

    if src and src in subs:
        u = _json3_url(subs[src])
        if u:
            return _TrackRef("manual", src, u)
    if src:
        for k in sorted(subs):
            if k.split("-")[0] == src:
                u = _json3_url(subs[k])
                if u:
                    return _TrackRef("manual", k, u)
    # auto: ONLY exact source code (ASR original) — avoids translated tracks
    if src and src in auto:
        u = _json3_url(auto[src])
        if u:
            return _TrackRef("auto", src, u)
    if not src and subs:
        k = sorted(subs)[0]
        u = _json3_url(subs[k])
        if u:
            return _TrackRef("manual", k, u)
    return None


def _parse_manual(data: dict) -> list[Segment]:
    out: list[Segment] = []
    for ev in data.get("events") or []:
        segs = ev.get("segs")
        if not segs:
            continue
        text = re.sub(r"\s+", " ", "".join(s.get("utf8", "") for s in segs)).strip()
        if not text:
            continue
        start = (ev.get("tStartMs") or 0) / 1000.0
        dur = (ev.get("dDurationMs") or 0) / 1000.0
        out.append(Segment(id=len(out), start=start, end=start + dur, text=text))
    return out


def _parse_auto_words(data: dict) -> list[_Word]:
    raw: list[tuple[str, float]] = []
    for ev in data.get("events") or []:
        segs = ev.get("segs")
        if not segs:
            continue
        base = ev.get("tStartMs") or 0
        for s in segs:
            w = (s.get("utf8") or "").strip()
            if not w:
                continue
            abs_start = (base + (s.get("tOffsetMs") or 0)) / 1000.0
            raw.append((w, abs_start))
    raw.sort(key=lambda x: x[1])

    deduped: list[tuple[str, float]] = []
    for text, t in raw:
        if deduped and deduped[-1][0].casefold() == text.casefold() and abs(t - deduped[-1][1]) < 0.10:
            continue
        deduped.append((text, t))

    # End of a word ≈ start of the next, but capped at _MAX_WORD_DUR so a silent
    # gap before the next word survives as a real pause (used to cut segments).
    words: list[_Word] = []
    for i, (text, t) in enumerate(deduped):
        if i + 1 < len(deduped):
            end = min(deduped[i + 1][1], t + _MAX_WORD_DUR)
        else:
            end = t + 0.30
        words.append(_Word(text=text, start=t, end=max(end, t)))
    return words


def _segment_auto_words(words: list[_Word], max_pause: float = 0.5,
                        max_words: int = 20) -> list[Segment]:
    """Group word-level auto-caption tokens into short segments at speech pauses.

    Cut at a silent gap (> ``max_pause``) or after ``max_words`` words. Each piece
    is capitalized and gets a trailing period so ``merge_continuous_segments`` (which
    otherwise merges lowercase-leading fragments) keeps the pieces separate; the
    translation step then punctuates the target text. Timestamps stay word-level.
    """
    segs: list[Segment] = []
    cur: list[_Word] = []

    def _flush() -> None:
        if not cur:
            return
        text = " ".join(w.text for w in cur).strip()
        if not text:
            return
        text = text[0].upper() + text[1:]
        if text[-1] not in ".!?…":
            text += "."
        segs.append(Segment(id=len(segs), start=cur[0].start, end=cur[-1].end, text=text))

    for i, w in enumerate(words):
        cur.append(w)
        gap = (words[i + 1].start - w.end) if i + 1 < len(words) else None
        if len(cur) >= max_words or (gap is not None and gap > max_pause):
            _flush()
            cur = []
    _flush()
    return segs


def _open_ydl():
    import yt_dlp

    opts = {"skip_download": True, "quiet": True, "no_warnings": True,
            "writesubtitles": True, "writeautomaticsub": True, "noplaylist": True}
    return yt_dlp.YoutubeDL(opts)


def _download_json3(url: str, *, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_source_captions(url: str, source_language: str = "auto-detect") -> list[Segment] | None:
    """Return Segment[] from the URL's source-language captions, or None.

    None means no usable caption was found — the caller should run Whisper.
    """
    try:
        with _open_ydl() as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        print(f"      [captions] extract_info failed: {exc}")
        return None

    src = _source_lang_code(info, source_language)
    track = _select_track(info, src)
    if track is None:
        print("      [captions] no suitable source-language caption track")
        return None

    try:
        data = _download_json3(track.url)
    except Exception as exc:
        print(f"      [captions] json3 download failed: {exc}")
        return None

    if track.kind == "manual":
        segs = _parse_manual(data)
    else:
        segs = _segment_auto_words(_parse_auto_words(data))

    if not segs:
        return None
    print(f"      [captions] using {track.kind} captions [{track.lang}] — {len(segs)} segments")
    return segs
