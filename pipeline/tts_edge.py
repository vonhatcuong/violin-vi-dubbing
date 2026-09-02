"""Edge-TTS backend.

Uses Microsoft Edge's public neural voices via the ``edge-tts`` package. The
pipeline only sends text for TTS synthesis; video/audio files remain local.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config as _conf
from .costs import CostTracker
from .ffmpeg_utils import FFMPEG_EXE
from .transcriber import Segment

_NATIVE_VOICES: dict[str, dict[str, dict[str, str]]] = {
    "vi": {
        "vi-VN-NamMinhNeural": {
            "gender": "male",
            "description": "Vietnamese male voice for narration and explainers.",
        },
        "vi-VN-HoaiMyNeural": {
            "gender": "female",
            "description": "Vietnamese female voice for narration and explainers.",
        },
    },
    "en": {
        "en-US-GuyNeural": {
            "gender": "male",
            "description": "US English male voice.",
        },
        "en-US-JennyNeural": {
            "gender": "female",
            "description": "US English female voice.",
        },
    },
}


def native_voices_for(language_code: str) -> list[str]:
    voices = _NATIVE_VOICES.get(language_code) or _NATIVE_VOICES["en"]
    names = list(voices.keys())
    male = next((n for n in names if voices[n]["gender"] == "male"), names[0])
    female = next((n for n in names if voices[n]["gender"] == "female"), names[-1])
    return [male, female]


def all_voices() -> dict[str, list[str]]:
    return {lang: list(voices.keys()) for lang, voices in _NATIVE_VOICES.items()}


def voice_descriptions() -> dict[str, str]:
    out: dict[str, str] = {}
    for voices in _NATIVE_VOICES.values():
        for name, meta in voices.items():
            out[name] = f"{meta['gender']} - {meta['description']}"
    return out


def _speed_to_rate(speed: float | None) -> str:
    if speed is None:
        return "+0%"
    pct = round((speed - 1.0) * 100)
    pct = max(-50, min(100, pct))
    return f"{pct:+d}%"


def _to_wav(mp3_path: str, wav_path: str, tail_ms: int) -> None:
    af = []
    if tail_ms > 0:
        af = ["-af", f"apad=pad_dur={tail_ms / 1000:.3f}"]
    subprocess.run(
        [FFMPEG_EXE, "-y", "-i", mp3_path,
         *af,
         "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "1",
         wav_path],
        check=True, capture_output=True,
    )


async def _save_mp3(text: str, voice: str, rate: str, mp3_path: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    await communicate.save(mp3_path)


def synthesize_segment(
    text: str,
    voice: str,
    output_path: str,
    client=None,
    language: str = "vi",
    speed: float | None = None,
    emotion: str | None = None,
) -> str:
    mp3_path = output_path + ".tmp.mp3"
    asyncio.run(_save_mp3(text, voice, _speed_to_rate(speed), mp3_path))

    tcfg = _conf.get().get("tts", {})
    if re.search(r'[.!?。！？]\s*$', text):
        tail_ms = tcfg.get("sentence_tail_silence_ms", tcfg.get("tail_silence_ms", 0))
    else:
        tail_ms = tcfg.get("tail_silence_ms", 0)
    _to_wav(mp3_path, output_path, tail_ms)
    Path(mp3_path).unlink(missing_ok=True)
    return output_path


def synthesize_segments(
    segments: list[Segment],
    voice: str,
    output_dir: str,
    client=None,
    language: str = "vi",
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

    done_count = 0
    with ThreadPoolExecutor(max_workers=_conf.get()["tts"]["workers"]) as pool:
        futures = {pool.submit(_do, i, seg): i for i, seg in enumerate(segments)}
        for future in as_completed(futures):
            idx, path = future.result()
            paths[idx] = path
            done_count += 1
            if done_count % 10 == 0 or done_count == total:
                print(f"      TTS progress: {done_count}/{total} segments done")

    return paths
