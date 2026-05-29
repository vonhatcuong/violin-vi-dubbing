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
