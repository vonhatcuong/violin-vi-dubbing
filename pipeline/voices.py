"""Fixed Vietnamese voice bank for zero-shot TTS backends (VieNeu-TTS, optional voice cloning).

A voice = reference clip (5–10 s, 24 kHz mono WAV) + its exact transcript.
Catalog lives in `<bank>/catalog.yaml`:

    voices:
      - name: nam-1
        gender: male          # male | female
        wav: nam-1.wav        # relative to the bank directory
        ref_text: "câu đúng nội dung clip, viết thường"
        description: "Giọng nam miền Bắc, trầm"

Create entries with `uv run scripts/make_ref_clip.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from . import config as _conf

if TYPE_CHECKING:
    from .transcriber import Segment

_REPO_ROOT = Path(__file__).resolve().parent.parent

# VieNeu-TTS v3 Turbo's 20 preset voices → gender. Used by `assign_voices` to
# pick a gender-matched preset out of `voices.speaker_voices` when the config
# doesn't supply its own `voices.preset_genders` map.
PRESET_GENDERS: dict[str, str] = {
    "Phạm Tuyên": "male",
    "Ngọc Huyền": "female",
    "Minh Đức": "male",
    "Trúc Ly": "female",
    "Thái Sơn": "male",
    "Mai Anh": "female",
    "Adam": "male",
    "Quang Sơn": "male",
    "Ngọc Trân": "female",
    "Xuân Vĩnh": "male",
    "Minh Triết": "male",
    "Đức Trí": "male",
    "Thục Đoan": "female",
    "Thùy Dung": "female",
    "Mỹ Duyên": "female",
    "Kim Thanh": "female",
    "Thanh Bình": "male",
    "Ngọc Linh": "female",
    "Đoan Trang": "female",
    "Quỳnh Anh": "female",
}


@dataclass(frozen=True)
class Voice:
    name: str
    gender: str
    ref_wav: Path
    ref_text: str
    description: str = ""


def bank_dir(bank: Path | None = None) -> Path:
    if bank is not None:
        return Path(bank)
    configured = _conf.get().get("voices", {}).get("bank", "assets/voices/vi")
    p = Path(configured).expanduser()
    return p if p.is_absolute() else _REPO_ROOT / p


def load_catalog(bank: Path | None = None) -> dict[str, Voice]:
    root = bank_dir(bank)
    catalog = root / "catalog.yaml"
    if not catalog.exists():
        return {}
    data = yaml.safe_load(catalog.read_text(encoding="utf-8")) or {}
    out: dict[str, Voice] = {}
    for entry in data.get("voices", []) or []:
        out[entry["name"]] = Voice(
            name=entry["name"],
            gender=entry.get("gender", "male"),
            ref_wav=root / entry["wav"],
            ref_text=entry["ref_text"],
            description=entry.get("description", ""),
        )
    return out


def get_voice(name: str, bank: Path | None = None) -> Voice:
    catalog = load_catalog(bank)
    if name not in catalog:
        raise KeyError(f"voice '{name}' not in bank {bank_dir(bank)} (have: {sorted(catalog)})")
    return catalog[name]


def native_voices_for(language_code: str, bank: Path | None = None) -> list[str]:
    """Return [default_male, default_female] names from the bank."""
    _ = language_code  # bank is per-language by directory; only `vi` shipped for now
    catalog = load_catalog(bank)
    if not catalog:
        raise RuntimeError(
            f"Voice bank {bank_dir(bank)} is empty. Add a reference clip with\n"
            "  uv run scripts/make_ref_clip.py --source clip.wav --start 0 --end 8 --name nam-1 --gender male"
        )
    vcfg = _conf.get().get("voices", {})

    def _first(gender: str, preferred: str) -> str:
        if preferred and preferred in catalog:
            return preferred
        for v in catalog.values():
            if v.gender == gender:
                return v.name
        return next(iter(catalog))

    return [_first("male", vcfg.get("default_male", "")), _first("female", vcfg.get("default_female", ""))]


def all_voices(bank: Path | None = None) -> dict[str, list[str]]:
    return {"vi": list(load_catalog(bank))}


def voice_descriptions(bank: Path | None = None) -> dict[str, str]:
    return {v.name: f"{v.gender} — {v.description}" for v in load_catalog(bank).values()}


def assign_voices(
    speakers: list[str],
    default_voice: str,
    voice_map: dict[str, str] | None = None,
    genders: dict[str, str] | None = None,
    bank: Path | None = None,
    speaker_voices: list[str] | None = None,
    preset_genders: dict[str, str] | None = None,
    seed_voice: str | None = None,
) -> dict[str, str]:
    """speaker → voice name. Priority: explicit map > seed voice > detected gender > round-robin > default.

    - `voice_map[spk]` always wins.
    - `seed_voice`, if given, goes to the first speaker (in `speakers` order)
      not already covered by `voice_map` — this is how an explicit
      `--voice`/`opts.voice` survives once `speaker_voices` is populated
      (its round-robin/gender cursors would otherwise never fall through to
      `default_voice`, silently discarding the user's choice).
    - Known gender + `speaker_voices` given: a per-gender cursor cycles through
      the gender-matching subset of `speaker_voices` (via `preset_genders`,
      default `PRESET_GENDERS`) — e.g. three male speakers each get a
      different male preset in turn, rather than all collapsing onto the
      first match. If `speaker_voices` has no entry of that gender, falls
      back to the voice bank's male/female pick (`native_voices_for`).
    - Unknown gender: round-robins over all of `speaker_voices`, in order of
      first appearance (old behaviour — falls back to `default_voice` — when
      `speaker_voices` is None/empty).
    - Voices already claimed by `voice_map` or `seed_voice` are excluded from
      the round-robin/gender pools whenever that still leaves an option (a
      fully-claimed pool falls back to reusing entries rather than erroring).
    """
    voice_map = voice_map or {}
    genders = genders or {}
    pg = PRESET_GENDERS if preset_genders is None else preset_genders

    claimed = set(voice_map.values())
    if seed_voice:
        claimed.add(seed_voice)

    def _pool(entries: list[str]) -> list[str]:
        filtered = [v for v in entries if v not in claimed]
        return filtered if filtered else list(entries)

    full_pool = list(speaker_voices or [])
    male_pool = _pool([v for v in full_pool if pg.get(v) == "male"])
    female_pool = _pool([v for v in full_pool if pg.get(v) == "female"])
    rr_pool = _pool(full_pool)

    bank_defaults: list[str] | None = None  # lazy — only touch the (often-empty) bank catalog if actually needed
    male_i = female_i = rr_i = 0
    seed_assigned = False

    out: dict[str, str] = {}
    for spk in speakers:
        if spk in voice_map:
            out[spk] = voice_map[spk]
            continue
        if seed_voice and not seed_assigned:
            out[spk] = seed_voice
            seed_assigned = True
            continue
        gender = genders.get(spk)
        if gender in ("male", "female"):
            pool = male_pool if gender == "male" else female_pool
            if pool:
                idx = male_i if gender == "male" else female_i
                out[spk] = pool[idx % len(pool)]
                if gender == "male":
                    male_i += 1
                else:
                    female_i += 1
            else:
                if bank_defaults is None:
                    try:
                        bank_defaults = native_voices_for("vi", bank=bank)
                    except RuntimeError:
                        bank_defaults = (default_voice, default_voice)
                male, female = bank_defaults
                out[spk] = female if gender == "female" else male
        elif rr_pool:
            out[spk] = rr_pool[rr_i % len(rr_pool)]
            rr_i += 1
        else:
            out[spk] = default_voice
    return out


def guess_genders(audio_path: str, segments: list["Segment"]) -> dict[str, str]:
    """Median-F0 gender guess per speaker.

    Concatenates up to 12 s of 16 kHz audio per speaker (from their segments),
    estimates F0 per 40 ms frame (640 samples, 320-sample hop) via normalized
    autocorrelation — a frame counts as voiced when RMS > 0.01 and the
    autocorrelation peak (searched over lag 40-228 samples, i.e. ~70-400 Hz)
    exceeds 0.5 — and classifies the speaker by the median voiced F0:
    < 165 Hz → male, else female. Speakers with no voiced frames are omitted.
    """
    import soundfile as sf

    wav, sr = sf.read(audio_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    n_samples = len(wav)
    max_samples = int(round(12.0 * sr))

    order: list[str] = []
    chunks: dict[str, list[np.ndarray]] = {}
    totals: dict[str, int] = {}
    for seg in segments:
        spk = seg.speaker
        if spk not in chunks:
            chunks[spk] = []
            totals[spk] = 0
            order.append(spk)
        remaining = max_samples - totals[spk]
        if remaining <= 0:
            continue
        s0 = max(0, min(n_samples, int(round(seg.start * sr))))
        s1 = max(0, min(n_samples, int(round(seg.end * sr))))
        if s1 <= s0:
            continue
        piece = wav[s0:s1][:remaining]
        chunks[spk].append(piece)
        totals[spk] += len(piece)

    frame, hop = 640, 320
    lag_min, lag_max = int(sr / 400), int(sr / 70)  # ~70-400 Hz search band, at 16 kHz: 40, 228
    out: dict[str, str] = {}
    for spk in order:
        pieces = chunks.get(spk) or []
        if not pieces:
            continue
        audio = np.concatenate(pieces)
        f0s: list[float] = []
        for start in range(0, len(audio) - frame + 1, hop):
            frm = audio[start:start + frame]
            rms = float(np.sqrt(np.mean(frm.astype(np.float64) ** 2)))
            if rms <= 0.01:
                continue
            centered = frm - frm.mean()
            ac = np.correlate(centered, centered, mode="full")
            ac = ac[len(ac) // 2:]
            if ac[0] <= 0:
                continue
            ac_norm = ac / ac[0]
            window = ac_norm[lag_min:lag_max + 1]
            if len(window) == 0:
                continue
            peak_i = int(np.argmax(window))
            peak_val = float(window[peak_i])
            if peak_val <= 0.5:
                continue
            lag = lag_min + peak_i
            f0 = sr / lag
            if 70.0 <= f0 <= 400.0:
                f0s.append(f0)
        if not f0s:
            continue
        median_f0 = float(np.median(f0s))
        out[spk] = "male" if median_f0 < 165.0 else "female"
    return out
