"""Local Vietnamese TTS backend: VieNeu-TTS v3 Turbo via the `vieneu` SDK.

Apache-2.0, 48 kHz, 20 preset voices (North / Central / South) — no reference
clip needed. On CPU (incl. Apple Silicon) the SDK runs ONNX int8 without
PyTorch; on CUDA it switches to a batched PyTorch engine (install
`torch==2.8.0` cu128 + `transformers==4.57.6` first, see README).

Voice names are either VieNeu presets ("Phạm Tuyên", "Ngọc Huyền", …) or
voice-bank entries (pipeline.voices) that get enrolled once via
`engine.add_voice`. Contract mirrors pipeline/tts_supertonic.py so
pipeline/tts.py can dispatch.
"""

from __future__ import annotations

import subprocess
import threading
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from . import config as _conf
from . import voices as _voices
from .costs import CostTracker
from .ffmpeg_utils import FFMPEG_EXE
from .transcriber import Segment
from .tts_supertonic import _append_silence
from .vi_text import normalize_for_tts

_LOCK = threading.Lock()
_ENGINE: Any = None
_ENROLLED: set[str] = set()   # bank voices already registered with the engine

_DEFAULT_MALE = "Phạm Tuyên"
_DEFAULT_FEMALE = "Ngọc Huyền"


def _vcfg() -> dict[str, Any]:
    return _conf.get().get("vieneu", {})


def get_shared_tts() -> Any:
    """Lazy-load one `vieneu.Vieneu` engine (downloads ~1 GB on first run)."""
    global _ENGINE
    if _ENGINE is None:
        with _LOCK:
            if _ENGINE is None:
                from vieneu import Vieneu
                cfg = _vcfg()
                print("      [vieneu] loading VieNeu-TTS v3 Turbo…")
                _ENGINE = Vieneu(
                    backend=cfg.get("backend", "auto"),
                    precision=cfg.get("precision", "int8"),
                    device=cfg.get("device", "auto"),
                    threads=int(cfg.get("threads", 0)),
                    max_batch_size=int(cfg.get("max_batch_size", 32)),
                )
    return _ENGINE


def unload() -> None:
    global _ENGINE
    with _LOCK:
        _ENGINE = None
        _ENROLLED.clear()
    from .devices import free_memory
    free_memory()


# ── voice catalog ───────────────────────────────────────────

def _preset_names(engine: Any | None = None) -> list[str]:
    if engine is None:
        return []
    try:
        return [voice_id for _label, voice_id in engine.list_preset_voices()]
    except Exception:
        return []


def native_voices_for(language_code: str) -> list[str]:
    """[male, female] preset names from config (defaults: Phạm Tuyên / Ngọc Huyền)."""
    _ = language_code  # vi only
    vcfg = _conf.get().get("voices", {})
    return [vcfg.get("default_male") or _DEFAULT_MALE, vcfg.get("default_female") or _DEFAULT_FEMALE]


def all_voices() -> dict[str, list[str]]:
    names = list(_voices.load_catalog())
    return {"vi": [_DEFAULT_MALE, _DEFAULT_FEMALE] + [n for n in names if n not in (_DEFAULT_MALE, _DEFAULT_FEMALE)]}


def voice_descriptions() -> dict[str, str]:
    out = {_DEFAULT_MALE: "male — VieNeu preset (Bắc, natural)", _DEFAULT_FEMALE: "female — VieNeu preset (Bắc, natural)"}
    out.update(_voices.voice_descriptions())
    return out


def _resolve_voice(engine: Any, voice: str) -> str:
    """Return the name to pass to `engine.infer(voice=...)`, enrolling bank clips once."""
    voice = unicodedata.normalize("NFC", voice)
    presets = _preset_names(engine)
    if voice in presets:
        return voice
    catalog = _voices.load_catalog()
    if voice in catalog:
        if voice not in _ENROLLED:
            engine.add_voice(voice, str(catalog[voice].ref_wav))
            _ENROLLED.add(voice)
        return voice
    if not presets:  # engine gave no list (fake / old SDK): trust the caller
        return voice
    raise KeyError(f"voice '{voice}' is neither a VieNeu preset {presets} nor in the voice bank {sorted(catalog)}")


# ── synthesis ───────────────────────────────────────────────

def _write_44k_mono(wav: np.ndarray, sr: int, output_path: str) -> None:
    tmp = output_path + ".raw.wav"
    sf.write(tmp, np.asarray(wav, dtype=np.float32), sr, subtype="PCM_16")
    subprocess.run(
        [FFMPEG_EXE, "-y", "-v", "error", "-i", tmp, "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", output_path],
        check=True, capture_output=True,
    )
    Path(tmp).unlink(missing_ok=True)


def _prepare_text(text: str, language: str, cfg: dict[str, Any]) -> str:
    text = unicodedata.normalize("NFC", text or "")
    if not language.lower().startswith("vi") or not cfg.get("normalize_numbers", False):
        return text
    return normalize_for_tts(text, use_vinorm=False, loanwords=cfg.get("loanwords") or {}, lowercase=False)


def _tail_ms(text: str, tcfg: dict[str, Any]) -> float:
    """Tail silence duration for one text: sentence-end punctuation gets the longer pause."""
    if text.rstrip().endswith((".", "!", "?", "。", "！", "？")):
        return tcfg.get("sentence_tail_silence_ms", tcfg.get("tail_silence_ms", 0))
    return tcfg.get("tail_silence_ms", 0)


def synthesize_segment(
    text: str,
    voice: str,
    output_path: str,
    client: Any,
    language: str = "vi",
    speed: float | None = None,
    emotion: str | None = None,
) -> str:
    """Synthesize one segment; write 44.1 kHz mono WAV with tail silence."""
    _ = speed, emotion  # VieNeu has neither speed nor emotion parameters
    engine = client if client is not None else get_shared_tts()
    cfg = _vcfg()
    gen_text = _prepare_text(text, language, cfg)
    with _LOCK:
        name = _resolve_voice(engine, voice)
        wav = engine.infer(
            gen_text,
            voice=name,
            temperature=float(cfg.get("temperature", 0.8)),
            top_k=int(cfg.get("top_k", 25)),
            top_p=float(cfg.get("top_p", 0.95)),
            repetition_penalty=float(cfg.get("repetition_penalty", 1.2)),
            apply_watermark=bool(cfg.get("watermark", True)),
        )
    _write_44k_mono(wav, int(getattr(engine, "sample_rate", 48000)), output_path)

    tcfg = _conf.get().get("tts", {})
    _append_silence(output_path, _tail_ms(text, tcfg))
    return output_path


def synthesize_batch(
    texts: list[str], voice: str, output_paths: list[str], client: Any, language: str = "vi",
) -> list[str]:
    """Synthesize many segments in one engine call (GPU static batching); same output format as synthesize_segment."""
    if len(texts) != len(output_paths):
        raise ValueError("texts and output_paths must have the same length")
    if not texts:
        return []
    engine = client if client is not None else get_shared_tts()
    cfg = _vcfg()
    prepared = [_prepare_text(t, language, cfg) for t in texts]
    with _LOCK:
        name = _resolve_voice(engine, voice)
        wavs = engine.infer_batch(
            prepared,
            voice=name,
            temperature=float(cfg.get("temperature", 0.8)),
            top_k=int(cfg.get("top_k", 25)),
            top_p=float(cfg.get("top_p", 0.95)),
            repetition_penalty=float(cfg.get("repetition_penalty", 1.2)),
            apply_watermark=bool(cfg.get("watermark", True)),
        )
    sr = int(getattr(engine, "sample_rate", 48000))
    tcfg = _conf.get().get("tts", {})
    for text, wav, path in zip(texts, wavs, output_paths):
        _write_44k_mono(wav, sr, path)
        _append_silence(path, _tail_ms(text, tcfg))
    return list(output_paths)


def synthesize_segments(
    segments: list[Segment],
    voice: str,
    output_dir: str,
    client: Any,
    language: str = "vi",
    voice_map: dict[str, str] | None = None,
    tracker: CostTracker | None = None,
    speed: float | None = None,
    emotion: str | None = None,
) -> list[str]:
    """Serial synthesis (the engine is not thread-safe; GPU batching is internal to the SDK)."""
    vm = voice_map or {}
    paths: list[str] = []
    for i, seg in enumerate(segments):
        path = str(Path(output_dir) / f"seg_{seg.id:05d}.wav")
        synthesize_segment(seg.text, vm.get(seg.speaker, voice), path, client, language, speed, emotion)
        if tracker:
            tracker.add_tts_usage(len(seg.text))
        paths.append(path)
        if (i + 1) % 10 == 0 or i + 1 == len(segments):
            print(f"      TTS progress: {i + 1}/{len(segments)} segments done")
    return paths
