"""Fixed Vietnamese voice bank for zero-shot TTS backends (F5-TTS-vi).

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

import yaml

from . import config as _conf

_REPO_ROOT = Path(__file__).resolve().parent.parent


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
) -> dict[str, str]:
    """speaker → voice name. Priority: explicit map > detected gender > default."""
    voice_map = voice_map or {}
    genders = genders or {}
    male, female = native_voices_for("vi", bank=bank)
    out: dict[str, str] = {}
    for spk in speakers:
        if spk in voice_map:
            out[spk] = voice_map[spk]
        elif genders.get(spk) == "female":
            out[spk] = female
        elif genders.get(spk) == "male":
            out[spk] = male
        else:
            out[spk] = default_voice
    return out
