"""Local Whisper backend via faster-whisper (CTranslate2).

Exposes an OpenAI/Together-compatible surface so the rest of the pipeline
(`pipeline.transcriber.transcribe`) does not need to branch.

The adapter implements `.audio.transcriptions.create(file, model, ...)`
returning an object with `.words`, `.segments`, `.text` attributes that match
what the cloud Whisper APIs return — same shape `_transcribe_single` expects.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any


# Cache the loaded model across calls — load is expensive (~3GB for large-v3).
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def _pick_device(requested: str) -> tuple[str, str]:
    """Resolve (device, compute_type) when user requests 'auto'.

    Apple Silicon: CPU + int8 is the sweet spot — CTranslate2 has very fast
    AVX/NEON kernels, and CoreML/MPS backend is not yet wired into CT2 stably.
    """
    if requested != "auto":
        # Caller specified explicitly — trust them, but pick a reasonable
        # compute_type default.
        return requested, "int8" if requested == "cpu" else "float16"

    machine = platform.machine().lower()
    system = platform.system().lower()
    if system == "darwin" and machine in ("arm64", "aarch64"):
        return "cpu", "int8"
    # Linux/Windows: try CUDA if available, else CPU.
    try:
        import ctranslate2  # noqa: F401
        # CTranslate2's device discovery is internal; cheapest probe is to
        # just attempt CUDA and fall back. We default to CPU here for safety
        # — users with GPU can override via config.transcription.local_device.
        return "cpu", "int8"
    except Exception:
        return "cpu", "int8"


def _resolve_compute_type(device: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "int8" if device == "cpu" else "float16"


def _load_model(model_size: str, device: str, compute_type: str) -> Any:
    """Lazy-load WhisperModel; cache by (size, device, compute_type) tuple."""
    key = (model_size, device, compute_type)
    if key not in _MODEL_CACHE:
        from faster_whisper import WhisperModel

        print(f"      [faster-whisper] loading {model_size} "
              f"(device={device}, compute_type={compute_type})…")
        _MODEL_CACHE[key] = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
    return _MODEL_CACHE[key]


@dataclass
class _Word:
    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class _Segment:
    id: int
    start: float
    end: float
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0


@dataclass
class _Response:
    text: str
    words: list[_Word]
    segments: list[_Segment]
    language: str | None = None


class _TranscriptionsClient:
    """Implements `.create(file, model, response_format, ...)` like the OpenAI SDK."""

    def __init__(self, model: Any):
        self._model = model

    def create(
        self,
        *,
        file: Any,
        model: str | None = None,
        response_format: str | None = None,
        timestamp_granularities: list[str] | None = None,
        timeout: float | None = None,
        **_ignored: Any,
    ) -> _Response:
        # `file` may be a path string, a (name, fileobj) tuple, or a file-like.
        audio_arg = _coerce_file_arg(file)
        word_ts = bool(timestamp_granularities and "word" in timestamp_granularities)

        segments_iter, info = self._model.transcribe(
            audio_arg,
            word_timestamps=word_ts,
            vad_filter=True,
            beam_size=5,
        )

        all_words: list[_Word] = []
        all_segments: list[_Segment] = []
        full_text_parts: list[str] = []

        for seg in segments_iter:
            seg_text = (seg.text or "").strip()
            all_segments.append(_Segment(
                id=len(all_segments),
                start=float(seg.start),
                end=float(seg.end),
                text=seg_text,
                no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
                avg_logprob=float(getattr(seg, "avg_logprob", 0.0) or 0.0),
            ))
            full_text_parts.append(seg_text)
            if word_ts and getattr(seg, "words", None):
                for w in seg.words:
                    all_words.append(_Word(
                        word=w.word,
                        start=float(w.start),
                        end=float(w.end),
                        probability=float(getattr(w, "probability", 1.0) or 1.0),
                    ))

        return _Response(
            text=" ".join(full_text_parts).strip(),
            words=all_words,
            segments=all_segments,
            language=getattr(info, "language", None),
        )


def _coerce_file_arg(file: Any) -> str:
    """Reduce the various file shapes Violin passes in to a path str.

    faster-whisper accepts a path string or a numpy waveform; the simplest
    universal route here is to write file-like inputs through a temp path,
    but Violin only ever calls us with `(name, fileobj)` where fileobj is
    an open binary file pointed at a real wav on disk — so just use the
    fileobj's `.name` attribute when available.
    """
    if isinstance(file, str):
        return file
    if isinstance(file, tuple) and len(file) >= 2:
        name = file[0]
        fobj: Any = file[1]
        path = getattr(fobj, "name", None)
        if isinstance(path, str):
            return path
        # Fall back: dump the bytes to a temp file.
        import tempfile
        suffix = "_" + (str(name) if name else "audio.wav")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            fobj.seek(0)
            tmp.write(fobj.read())
            return tmp.name
    path = getattr(file, "name", None)
    if isinstance(path, str):
        return path
    raise TypeError(f"Unsupported file argument for faster-whisper: {type(file)!r}")


class _AudioNamespace:
    def __init__(self, model: Any):
        self.transcriptions = _TranscriptionsClient(model)


class FasterWhisperAdapter:
    """Drop-in replacement for an OpenAI/Together client when transcription
    runs locally via faster-whisper.

    Only the `audio.transcriptions.create(...)` surface is implemented — that
    is all `pipeline.transcriber._transcribe_single` ever calls on the client.
    """

    def __init__(
        self,
        *,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "auto",
    ):
        resolved_device, _default_ct = _pick_device(device)
        resolved_ct = _resolve_compute_type(resolved_device, compute_type)
        model = _load_model(model_size, resolved_device, resolved_ct)
        self.audio = _AudioNamespace(model)
