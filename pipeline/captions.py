"""Fetch & normalize source captions (YouTube etc.) into Segment[].

When a video URL already ships captions we prefer them over re-running Whisper:
faster, cheaper, and often more accurate for proper nouns. Manual captions are
used as-is (they carry punctuation); automatic captions are word-level but
unpunctuated, so an LLM restores punctuation and we re-align the punctuated text
back onto the original word timestamps. Any failure falls back to Whisper.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass

from . import config as _conf
from .costs import CostTracker
from .languages import language_code
from .llm_client import get_translation_model, get_translation_provider, make_translation_client
from .transcriber import Segment, _is_sentence_end

import prompts as _prompts

_SENT_END = re.compile(r"[.!?…。！？]+$")
_PUNCT_STRIP = re.compile(r"^[\W_]+|[\W_]+$")  # strip leading/trailing non-word (unicode-aware)

_PUNCT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


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

    words: list[_Word] = []
    for i, (text, t) in enumerate(deduped):
        end = deduped[i + 1][1] if i + 1 < len(deduped) else t + 0.30
        words.append(_Word(text=text, start=t, end=max(end, t)))
    return words


def _norm(tok: str) -> str:
    return _PUNCT_STRIP.sub("", tok).casefold()


def _chunk_words(words: list[_Word], max_words: int = 120, max_gap: float = 2.0) -> list[list[_Word]]:
    chunks: list[list[_Word]] = []
    cur: list[_Word] = []
    for i, w in enumerate(words):
        cur.append(w)
        gap_break = i + 1 < len(words) and (words[i + 1].start - w.end) > max_gap
        if len(cur) >= max_words or gap_break:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def _align_chunk(words: list[_Word], punctuated: str) -> list[Segment] | None:
    if not words:
        return []
    tokens = punctuated.split()
    if not tokens:
        return None
    if abs(len(tokens) - len(words)) / max(1, len(words)) > 0.15:
        return None  # LLM altered the word stream → caller falls back

    segs: list[Segment] = []
    cur_words: list[_Word] = []
    cur_text: list[str] = []
    wi = 0
    for tok in tokens:
        if wi < len(words):
            cur_words.append(words[wi])
            wi += 1
        cur_text.append(tok)
        if _SENT_END.search(tok) and _is_sentence_end(tok):
            text = " ".join(cur_text).strip()
            if cur_words and text:
                segs.append(Segment(id=len(segs), start=cur_words[0].start,
                                    end=cur_words[-1].end, text=text))
            cur_words, cur_text = [], []
    if cur_words and cur_text:
        text = " ".join(cur_text).strip()
        if text:
            segs.append(Segment(id=len(segs), start=cur_words[0].start,
                                end=cur_words[-1].end, text=text))
    return segs


def _together_extra() -> dict:
    if get_translation_provider(_conf.get()) == "together":
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {}


def _restore_punctuation(text: str, client, source_language: str,
                         tracker: CostTracker | None = None) -> str:
    cfg = _conf.get()
    model = get_translation_model(cfg)
    max_retries = cfg["translation"].get("max_retries", 3)
    system_msg = _prompts.load("restore_punctuation", "system", source_language=source_language)
    user_msg = _prompts.load("restore_punctuation", "user",
                             source_language=source_language, text=text)
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "restore_punctuation",
                                    "strict": True, "schema": _PUNCT_SCHEMA},
                },
                **_together_extra(),
            )
            if tracker and getattr(response, "usage", None):
                tracker.add_llm_usage(response.usage.prompt_tokens or 0,
                                      response.usage.completion_tokens or 0)
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)["text"]
        except Exception:  # transient API / parse error
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                raise
    raise RuntimeError("punctuation restore exhausted retries")
