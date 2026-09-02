"""Measure seconds-per-syllable of a voice-bank voice at TTS speed 1.0.

    uv run scripts/calibrate_voice.py --voice nam-1 --config config/local_mac.yaml
Prints the value to put in `fit.sec_per_syllable`.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config as pipeline_config  # noqa: E402
from pipeline.fitter import wav_duration  # noqa: E402
from pipeline.tts import make_synthesizer  # noqa: E402
from pipeline.vi_text import count_syllables  # noqa: E402

SENTENCES = [
    "hôm nay chúng ta sẽ tìm hiểu cách mô hình ngôn ngữ lớn học từ dữ liệu văn bản.",
    "trước hết, hãy nhìn vào cấu trúc của một mạng nơ ron đơn giản.",
    "kết quả cho thấy phương pháp mới nhanh hơn khoảng hai lần so với cách cũ.",
    "nếu bạn có câu hỏi, hãy để lại bình luận bên dưới video này.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--config", default="config/local_mac.yaml")
    args = ap.parse_args()
    pipeline_config.load(args.config)
    tail_s = float(pipeline_config.get().get("tts", {}).get("sentence_tail_silence_ms", 250)) / 1000.0
    print(f"subtracting {tail_s:.3f}s tail silence per clip (tts.sentence_tail_silence_ms)")
    synth = make_synthesizer(language="vi")
    total_s, total_syl = 0.0, 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, text in enumerate(SENTENCES):
            path = synth(text, args.voice, f"{tmp}/c{i}.wav", 1.0)
            dur = max(wav_duration(path) - tail_s, 0.01)   # subtract sentence tail silence
            syl = count_syllables(text)
            total_s += dur
            total_syl += syl
            print(f"  {syl:3d} syll  {dur:5.2f}s  → {dur / syl:.3f} s/syll")
    print(f"\nfit.sec_per_syllable: {total_s / total_syl:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
