"""Vietnamese text frontend for local TTS (F5-TTS-Vietnamese expects lowercase NFC text).

normalize_for_tts:  vinorm (if installed & enabled) → loanword map → numbers/percent
                    → lowercase → drop unsupported symbols → collapse spaces.
count_syllables:    Vietnamese is monosyllabic and space-delimited, so after
                    normalization every alphanumeric token is one syllable.
"""

from __future__ import annotations

import re
import unicodedata

try:  # optional: vinorm ships a native binary that may not exist on every platform
    from vinorm import TTSnorm as _vinorm_norm
except Exception:  # pragma: no cover - depends on platform
    _vinorm_norm = None

from num2words import num2words

_VI_LETTERS = "a-zA-ZÀ-ỹ"
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_PERCENT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
_TOKEN_RE = re.compile(rf"[{_VI_LETTERS}0-9]+")
_ALLOWED_RE = re.compile(rf"[^{_VI_LETTERS}0-9\s.,!?;:'\-]")
_SPACES_RE = re.compile(r"\s+")


def _number_to_words(token: str) -> str:
    text = token.replace(",", ".")
    try:
        value = float(text) if "." in text else int(text)
        return num2words(value, lang="vi")
    except Exception:
        return token


def _expand_numbers(text: str) -> str:
    text = _PERCENT_RE.sub(lambda m: f"{_number_to_words(m.group(1))} phần trăm", text)
    return _NUMBER_RE.sub(lambda m: _number_to_words(m.group(0)), text)


def _apply_loanwords(text: str, loanwords: dict[str, str] | None) -> str:
    for src, dst in (loanwords or {}).items():
        text = re.sub(rf"(?<![{_VI_LETTERS}0-9]){re.escape(src)}(?![{_VI_LETTERS}0-9])", dst, text, flags=re.IGNORECASE)
    return text


def normalize_for_tts(
    text: str,
    *,
    use_vinorm: bool | None = None,
    loanwords: dict[str, str] | None = None,
) -> str:
    """Return TTS-ready Vietnamese text: NFC, lowercase, numbers spelled out."""
    text = unicodedata.normalize("NFC", text or "")
    text = _apply_loanwords(text, loanwords)
    want_vinorm = (_vinorm_norm is not None) if use_vinorm is None else use_vinorm
    if want_vinorm and _vinorm_norm is not None:
        try:
            text = _vinorm_norm(text, punc=False, unknown=False, lower=True, rule=False)
        except Exception:
            pass  # fall through to the built-in expansion
    text = _expand_numbers(text)
    text = text.lower()
    text = _ALLOWED_RE.sub(" ", text)
    text = _SPACES_RE.sub(" ", text).strip()
    return text


def count_syllables(text: str) -> int:
    """Count spoken syllables ≈ alphanumeric tokens after normalization."""
    return len(_TOKEN_RE.findall(normalize_for_tts(text, use_vinorm=False)))
