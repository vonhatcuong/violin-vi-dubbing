"""Vietnamese text frontend for local TTS (F5-TTS-Vietnamese expects lowercase NFC text).

normalize_for_tts:  NFC → loanword map → vinorm (if installed & enabled) → numbers/percent
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
_PERCENT_RE = re.compile(r"(?<![\w])(-?)(\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?|\d+)\s*%(?![\w])")
_NUMBER_RE = re.compile(r"(?<![\w])(-?)(\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)(?![\w])")
_TOKEN_RE = re.compile(rf"[{_VI_LETTERS}0-9]+")
_ALLOWED_RE = re.compile(rf"[^{_VI_LETTERS}0-9\s.,!?;:'\-]")
_SPACES_RE = re.compile(r"\s+")


def _spell_number(sign: str, body: str) -> str:
    """Convert number body to words, handling thousands-grouping vs decimals.
    
    Thousands: All separator groups are exactly 3 digits (1,000; 1.000.000; 2.500).
    Decimals: Any other digit[.,]digit pattern (2.5; 3.14; 0.75).
    """
    # Detect if all separator groups are exactly 3 digits (thousands grouping)
    if re.match(r"^\d{1,3}(?:[.,]\d{3})+$", body):
        # Thousands: strip separators and convert as integer
        body_int = body.replace(",", "").replace(".", "")
        try:
            result = num2words(int(body_int), lang="vi")
        except Exception:
            result = body
    elif re.search(r"[.,]", body):
        # Decimal: split into integer and fractional parts, read each fractional digit individually
        normalized = body.replace(",", ".")
        if "." in normalized:
            int_part, frac_part = normalized.split(".", 1)
            try:
                if int_part == "0" or int_part == "":
                    int_words = "không"
                else:
                    int_words = num2words(int(int_part), lang="vi")
                # Read each fractional digit individually
                frac_words = " ".join(num2words(int(d), lang="vi") for d in frac_part)
                result = f"{int_words} phẩy {frac_words}"
            except Exception:
                result = body
        else:
            result = body
    else:
        # Plain integer
        try:
            result = num2words(int(body), lang="vi")
        except Exception:
            result = body
    
    # Prepend "âm " if negative
    if sign == "-":
        result = f"âm {result}"
    
    return result


def _expand_numbers(text: str) -> str:
    """Expand numbers and percentages to Vietnamese words."""
    # Handle percent first (e.g., 2.5% → "hai phẩy năm phần trăm")
    def replace_percent(m):
        sign = m.group(1)
        body = m.group(2)
        number_words = _spell_number(sign, body)
        return f"{number_words} phần trăm"
    
    text = _PERCENT_RE.sub(replace_percent, text)
    
    # Then handle all other numbers (with optional sign)
    def replace_number(m):
        sign = m.group(1)
        body = m.group(2)
        return _spell_number(sign, body)
    
    text = _NUMBER_RE.sub(replace_number, text)
    return text


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
