"""Cut a reference clip for the voice bank and register it in catalog.yaml.

    uv run scripts/make_ref_clip.py --source talk.mp4 --start 12 --end 20 --name nam-1 --gender male
    uv run scripts/make_ref_clip.py --source me.wav --start 0 --end 8 --name nu-1 --gender female --ref-text "..."

Without --ref-text the clip is transcribed locally with faster-whisper (language vi).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.ffmpeg_utils import FFMPEG_EXE  # noqa: E402
from pipeline.vi_text import normalize_for_tts  # noqa: E402
from pipeline.voices import bank_dir  # noqa: E402


def _transcribe_vi(path: Path) -> str:
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(path), language="vi", beam_size=5)
    return " ".join(s.text.strip() for s in segments).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--gender", choices=["male", "female"], required=True)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--description", default="")
    ap.add_argument("--bank", default=None, help="voice bank dir (default: config voices.bank)")
    args = ap.parse_args()

    if not 3.0 <= args.end - args.start <= 15.0:
        ap.error("clip length must be 3–15 s (F5-TTS reference sweet spot is 5–10 s)")

    bank = bank_dir(Path(args.bank) if args.bank else None)
    bank.mkdir(parents=True, exist_ok=True)
    wav = bank / f"{args.name}.wav"
    subprocess.run([
        FFMPEG_EXE, "-y", "-v", "error",
        "-ss", str(args.start), "-to", str(args.end), "-i", args.source,
        "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(wav),
    ], check=True)

    ref_text = args.ref_text or _transcribe_vi(wav)
    ref_text = normalize_for_tts(ref_text, use_vinorm=False)

    catalog_path = bank / "catalog.yaml"
    data = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {}
    entries = [e for e in (data.get("voices") or []) if e.get("name") != args.name]
    entries.append({
        "name": args.name, "gender": args.gender, "wav": wav.name,
        "ref_text": ref_text, "description": args.description,
    })
    catalog_path.write_text(yaml.safe_dump({"voices": entries}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Added voice '{args.name}' → {wav}\n  ref_text: {ref_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
