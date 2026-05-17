"""Local TTS backend via Supertonic 3 (ONNX, on-device, MIT license).

Runs entirely on the local machine — no API key, no network after first model
download (~400MB cached at ~/.cache/supertonic3/).

10 multilingual voice styles (M1-M5, F1-F5) work across 31 languages including
Vietnamese (`vi`). Output is 44.1kHz mono WAV — matches what the merger expects
after a single ffmpeg pass to align sample format.

Thread-safety: a single `supertonic.TTS` instance is shared by all worker
threads via a module-level lock; ONNX runtime sessions are not guaranteed
thread-safe across all execution providers. Set ``tts.workers: 1`` in config
to skip the lock entirely (recommended on Apple Silicon — synthesis is already
faster than real-time on Neural Engine).
"""

from __future__ import annotations

import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import soundfile as sf

from . import config as _conf
from .costs import CostTracker
from .ffmpeg_utils import FFMPEG_EXE
from .transcriber import Segment


# Supertonic 3's built-in voice styles. They are multilingual — the same M1/F1
# work across all 31 supported languages, so we expose one catalog under "en"
# and let every language fall through to it (mirrors the OpenAI TTS pattern).
_VOICE_STYLES: dict[str, dict[str, str]] = {
    "M1": {"gender": "male",   "description": "Warm, conversational male — default narrator."},
    "M2": {"gender": "male",   "description": "Energetic male — engaging for tutorials and short-form."},
    "M3": {"gender": "male",   "description": "Calm, low-pitched male — long-form narration."},
    "M4": {"gender": "male",   "description": "Bright, youthful male — casual content."},
    "M5": {"gender": "male",   "description": "Authoritative male — news and serious content."},
    "F1": {"gender": "female", "description": "Friendly, conversational female — default narrator."},
    "F2": {"gender": "female", "description": "Bright, expressive female — engaging short-form."},
    "F3": {"gender": "female", "description": "Soft, gentle female — calm narration."},
    "F4": {"gender": "female", "description": "Energetic female — kids and casual content."},
    "F5": {"gender": "female", "description": "Authoritative female — news and formal."},
}

_DEFAULT_MALE = "M1"
_DEFAULT_FEMALE = "F1"


# Shared TTS instance + lock (the "client" returned from _make_client).
_TTS_LOCK = threading.Lock()
_TTS_INSTANCE: Any = None


def native_voices_for(language_code: str) -> list[str]:
    """Return [primary_male, primary_female]. Voices are multilingual."""
    _ = language_code  # unused — Supertonic voices are language-agnostic.
    return [_DEFAULT_MALE, _DEFAULT_FEMALE]


def all_voices() -> dict[str, list[str]]:
    return {"en": list(_VOICE_STYLES.keys())}


def voice_descriptions() -> dict[str, str]:
    return {name: f"{meta['gender']} — {meta['description']}"
            for name, meta in _VOICE_STYLES.items()}


def get_shared_tts() -> Any:
    """Lazy-load the global Supertonic TTS instance.

    Returned from `pipeline.tts._make_client` when provider == 'supertonic'.
    Loading downloads ~400MB to ~/.cache/supertonic3/ on first run.
    """
    global _TTS_INSTANCE
    if _TTS_INSTANCE is None:
        with _TTS_LOCK:
            if _TTS_INSTANCE is None:
                from supertonic import TTS
                print("      [supertonic] loading model (first run downloads ~400MB)…")
                _TTS_INSTANCE = TTS(auto_download=True)
    return _TTS_INSTANCE


def _supertonic_lang_code(language: str) -> str:
    """Map BCP-47 / Violin language codes to Supertonic's ISO codes.

    Supertonic 3 supports 31 codes (ar, bg, cs, da, de, el, en, es, et, fi, fr,
    hi, hr, hu, id, it, ja, ko, lt, lv, nl, pl, pt, ro, ru, sk, sl, sv, tr,
    uk, vi). Unsupported codes fall back to 'na' (language-agnostic).
    """
    supported = {
        "ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr",
        "hi", "hr", "hu", "id", "it", "ja", "ko", "lt", "lv", "nl", "pl",
        "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
    }
    code = (language or "en").lower()[:2]
    return code if code in supported else "na"


def _write_wav_44k_mono(wav_array: Any, output_path: str, sample_rate: int = 44100) -> None:
    """Write Supertonic's float32 numpy output to disk as PCM WAV.

    Supertonic emits shape (1, N) float32 @ 44.1 kHz. soundfile expects
    (N,) or (N, channels) — squeeze the leading singleton axis.
    """
    arr = wav_array
    if hasattr(arr, "ndim") and arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    sf.write(output_path, arr, sample_rate, subtype="PCM_16")


def _append_silence(path: str, ms: int) -> None:
    """Pad *ms* milliseconds of silence to the end of *path* via ffmpeg."""
    if ms <= 0:
        return
    tmp = path + ".pad.wav"
    subprocess.run(
        [FFMPEG_EXE, "-y", "-i", path,
         "-af", f"apad=pad_dur={ms / 1000:.3f}",
         "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "1",
         tmp],
        check=True, capture_output=True,
    )
    Path(tmp).replace(Path(path))


def _resolve_voice_style(tts: Any, voice: str) -> Any:
    """Look up a Supertonic voice style object by user-supplied name."""
    name = voice if voice in _VOICE_STYLES else _DEFAULT_MALE
    return tts.get_voice_style(voice_name=name)


# ── Expression tag injection ────────────────────────────────
# Supertonic 3 supports 10 inline tags: <laugh>, <breath>, <sigh>, etc.
# We inject <breath> heuristically to give long sentences natural pauses.

_LONG_SEGMENT_CHARS = 90                  # only inject for segments above this length
_BREATH_PUNCT_RE = re.compile(r'[,，;；:：]')


def _inject_breath_tags(text: str) -> str:
    """Insert a single ``<breath>`` after the comma/semicolon nearest the segment midpoint.

    Triggers only for segments longer than ``_LONG_SEGMENT_CHARS`` AND containing
    at least one clause-break punctuation mark. Conservative single-tag insertion
    avoids over-pausing short narration.
    """
    if len(text) < _LONG_SEGMENT_CHARS:
        return text
    matches = list(_BREATH_PUNCT_RE.finditer(text))
    if not matches:
        return text
    mid = len(text) / 2
    best = min(matches, key=lambda m: abs(m.end() - mid))
    pos = best.end()
    # Skip if already directly followed by an existing tag or whitespace+tag.
    rest = text[pos:].lstrip()
    if rest.startswith("<"):
        return text
    return text[:pos] + " <breath>" + text[pos:]


def synthesize_segment(
    text: str,
    voice: str,
    output_path: str,
    client: Any,
    language: str = "en",
    speed: float | None = None,
    emotion: str | None = None,
) -> str:
    """Synthesize one segment via Supertonic, write WAV, pad tail silence."""
    _ = emotion  # Supertonic 3 inline expression tags would go in `text`; not used here.
    tts = client  # client IS the shared Supertonic TTS instance.

    cfg = _conf.get()
    sup_cfg = cfg.get("supertonic", {})
    total_steps = sup_cfg.get("total_steps", 8)
    sup_speed = speed if speed is not None else sup_cfg.get("speed", 1.0)
    sup_speed = max(0.7, min(2.0, float(sup_speed)))

    style = _resolve_voice_style(tts, voice)
    lang = _supertonic_lang_code(language)

    if sup_cfg.get("inject_breath", False):
        text = _inject_breath_tags(text)

    # ONNX runtime sessions are not universally thread-safe — serialize calls.
    with _TTS_LOCK:
        wav, _duration = tts.synthesize(
            text=text,
            lang=lang,
            voice_style=style,
            total_steps=total_steps,
            speed=sup_speed,
        )

    _write_wav_44k_mono(wav, output_path)

    tcfg = cfg.get("tts", {})
    if re.search(r'[.!?。！？]\s*$', text):
        tail_ms = tcfg.get("sentence_tail_silence_ms", tcfg.get("tail_silence_ms", 0))
    else:
        tail_ms = tcfg.get("tail_silence_ms", 0)
    _append_silence(output_path, tail_ms)
    return output_path


def synthesize_segments(
    segments: list[Segment],
    voice: str,
    output_dir: str,
    client: Any,
    language: str = "en",
    voice_map: dict[str, str] | None = None,
    tracker: CostTracker | None = None,
    speed: float | None = None,
    emotion: str | None = None,
) -> list[str]:
    total = len(segments)
    paths = [""] * total
    vm = voice_map or {}

    def _do(idx: int, seg: Segment) -> tuple[int, str]:
        path = str(Path(output_dir) / f"seg_{seg.id:05d}.wav")
        seg_voice = vm.get(seg.speaker, voice)
        synthesize_segment(seg.text, seg_voice, path, client, language, speed, emotion)
        if tracker:
            tracker.add_tts_usage(len(seg.text))
        return idx, path

    workers = _conf.get()["tts"]["workers"]
    done_count = 0

    def _record(idx: int, path: str) -> None:
        nonlocal done_count
        paths[idx] = path
        done_count += 1
        if done_count % 10 == 0 or done_count == total:
            print(f"      TTS progress: {done_count}/{total} segments done")

    # Even when workers > 1, the synthesize() call itself is serialized inside
    # the global _TTS_LOCK. ThreadPool here mainly overlaps the ffmpeg silence-
    # padding step with the next synthesis call.
    if workers <= 1:
        for i, seg in enumerate(segments):
            idx, path = _do(i, seg)
            _record(idx, path)
        return paths

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_do, i, seg): i for i, seg in enumerate(segments)}
        for future in as_completed(futures):
            idx, path = future.result()
            _record(idx, path)

    return paths
