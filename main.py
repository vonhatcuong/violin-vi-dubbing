"""
Violin CLI

Usage:
    uv run main.py <input_video> <output_video> --language <target_language>

Examples:
    uv run main.py lecture.mp4 lecture_es.mp4 --language Spanish
    uv run main.py lesson.mp4 lesson_ja.mp4 --language Japanese
    uv run main.py talk.mp4 talk_zh.mp4 --language Chinese --style kids
"""

import argparse
import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

from pipeline import config as pipeline_config
from pipeline.captions import fetch_source_captions
from pipeline.downloader import download_url_to_file, is_url
from pipeline.orchestrator import DubOptions, dub_video, valid_speakers_value
from pipeline.styles import list_styles, resolve as resolve_style

load_dotenv(override=True)


def _print_styles() -> None:
    """Print available style profiles and exit."""
    styles = list_styles()
    if not styles:
        print("No styles defined in config.")
        return
    print("Available styles:\n")
    for s in styles:
        print(f"  {s.name:14s}  {s.description}")
        parts = []
        if s.tts_speed is not None:
            parts.append(f"speed={s.tts_speed}")
        if s.tts_emotion:
            parts.append(f"emotion={s.tts_emotion}")
        if parts:
            print(f"  {'':14s}  TTS: {', '.join(parts)}")


def _speakers_type(value: str) -> str:
    """argparse `type=` validator for `--speakers`: "auto" or a positive integer string (no leading zero)."""
    if not valid_speakers_value(value):
        raise argparse.ArgumentTypeError(f'--speakers must be "auto" or a positive integer, got {value!r}')
    return value


def _parse_voice_map(value: str | None) -> dict[str, str] | None:
    """argparse `type=` for `--voice-map`: parse `"SPEAKER_00=Phạm Tuyên,SPEAKER_01=Ngọc Huyền"` into a dict.

    Returns None for an empty/unset value. Raises `argparse.ArgumentTypeError`
    on a malformed entry (no "=", or an empty speaker/voice name).
    """
    if not value:
        return None
    out: dict[str, str] = {}
    for pair in value.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise argparse.ArgumentTypeError(f'--voice-map entry {pair!r} must be "SPEAKER=Voice Name"')
        speaker, _, name = pair.partition("=")
        speaker, name = speaker.strip(), name.strip()
        if not speaker or not name:
            raise argparse.ArgumentTypeError(f'--voice-map entry {pair!r} must be "SPEAKER=Voice Name"')
        out[speaker] = name
    return out or None


def _install_skill() -> None:
    """Copy the bundled Claude Code skill into ~/.claude/skills/ and exit."""
    import shutil
    src = Path(__file__).resolve().parent / ".claude" / "skills" / "video-translator"
    if not src.is_dir():
        sys.stderr.write(
            f"ERROR: bundled skill files not found at {src}\n"
            "       This usually means an older Violin release that predates "
            "the install-skill feature — upgrade with `uv tool install --pre --upgrade violin`.\n"
        )
        sys.exit(1)
    dst = Path.home() / ".claude" / "skills" / "video-translator"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"Installed Violin skill → {dst}")


def translate_video(
    input_path: str,
    output_path: str,
    target_language: str,
    voice: str | None = None,
    subtitles: bool = True,
    source_language: str = "auto-detect",
    style=None,
    voiceover: bool = True,
    subtitle_formats: tuple[str, ...] = ("srt",),
    burn_subtitles: bool = False,
    timings_out: str | None = None,
    prefer_source_captions: bool = True,
    fit: bool | None = None,
    subtitle_lang: str | None = None,
    speakers: str = "1",
    voice_map: dict[str, str] | None = None,
) -> None:
    if style is None:
        style = resolve_style("standard")

    if style.name != "standard":
        print(f"\n  Style: {style.name} — \"{style.description}\"")

    out_p = Path(output_path)
    srt_path = str(out_p.with_suffix(".srt")) if subtitles else None
    orig_audio_path = str(out_p.with_stem(out_p.stem + "_original").with_suffix(".m4a")) if voiceover else None

    opts = DubOptions(
        target_language=target_language,
        source_language=source_language,
        voice=voice,
        style=style,
        voiceover=voiceover,
        bake_voiceover=True,
        subtitles=subtitles,
        subtitle_lang=subtitle_lang,
        subtitle_formats=subtitle_formats,
        burn_subtitles=burn_subtitles,
        fit=fit,
        speakers=speakers,
        voice_map=voice_map,
    )

    tmp_download_dir = None
    try:
        effective_input = input_path
        segments_override = None
        if is_url(input_path):
            tmp_download_dir = tempfile.mkdtemp(prefix="violin_url_")
            print(f"\n[0/5] Downloading media URL…")
            effective_input = str(download_url_to_file(input_path, tmp_download_dir))
            if prefer_source_captions:
                print(f"\n[0/5] Checking for source captions…")
                segments_override = fetch_source_captions(input_path, source_language)

        result = dub_video(
            effective_input,
            output_path,
            opts,
            output_srt_path=srt_path,
            burned_video_path=str(out_p.with_stem(out_p.stem + "_subtitled")) if burn_subtitles else None,
            original_audio_path=orig_audio_path,
            on_progress=lambda step, msg: print(f"\n[{step}/5] {msg}"),
            segments_override=segments_override,
        )
    finally:
        if tmp_download_dir:
            shutil.rmtree(tmp_download_dir, ignore_errors=True)

    if result.original_audio_path:
        print(f"      Original audio → {result.original_audio_path}")
    for fmt, path in result.subtitle_paths.items():
        print(f"      Subtitles ({fmt}) → {path}")
    if result.burned_video_path:
        print(f"      Burned-subtitle video → {result.burned_video_path}")
    if result.transcript_path:
        print(f"      Transcript → {result.transcript_path}")

    print(f"\nDone! Output → {result.output_video_path}")
    result.cost_tracker.print_summary()

    if timings_out:
        payload = {
            "total": sum(s["elapsed"] for s in result.steps),
            "steps": result.steps,
            "cost": result.cost_tracker.cost_breakdown(),
        }
        Path(timings_out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"      Timings → {timings_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate a video to another language using Together AI."
    )
    parser.add_argument("input", nargs="?", help="Input media file path or URL")
    parser.add_argument("output", nargs="?", help="Output video file path")
    parser.add_argument(
        "--language", "-l", default=None,
        help="Target language (e.g. Spanish, French, Japanese, Arabic)"
    )
    parser.add_argument(
        "--voice", "-v", default=None,
        help="TTS voice for translated speech (default: pick native voice by preferences.voice_gender)"
    )
    parser.add_argument(
        "--source-language", default="auto-detect",
        help="Source language hint for translation (default: auto-detect)"
    )
    parser.add_argument(
        "--no-subtitles", action="store_true",
        help="Skip generating SRT subtitle file"
    )
    parser.add_argument(
        "--subtitle-formats", default="srt",
        help="Comma-separated subtitle formats to write: srt,vtt,txt (default: srt)"
    )
    parser.add_argument(
        "--subtitle-lang", choices=["source", "target"], default=None,
        help="Subtitle language: source = original sentences re-timed to the output video, "
             "target = translated text (default: config subtitles.language)"
    )
    parser.add_argument(
        "--burn-subtitles", action="store_true",
        help="Also write a second video with subtitles burned into the picture"
    )
    parser.add_argument(
        "--voiceover", action="store_true", default=None,
        help="Voice-over mode: keep original audio underneath the dub (default)"
    )
    parser.add_argument(
        "--no-voiceover", action="store_true", default=None,
        help="Full replacement: dubbed audio only, no original audio"
    )
    parser.add_argument(
        "--style", "-s", default=None,
        help='Translation style profile (e.g. standard, kids, academic, casual). '
             'Use "--style list" to see all available styles.'
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to a YAML config file (overrides config/default.yaml)"
    )
    parser.add_argument(
        "--timings-out", default=None,
        help="Write per-step wall-clock timings as JSON to this path on success"
    )
    parser.add_argument(
        "--no-source-captions", action="store_true",
        help="Always transcribe with Whisper instead of reusing the video's captions (URL input)"
    )
    parser.add_argument(
        "--install-skill", action="store_true",
        help="Copy the Violin Claude Code skill to ~/.claude/skills/ and exit"
    )
    parser.add_argument(
        "--no-fit", action="store_true",
        help="Disable the duration fitter even if the config enables it (local dubbing)",
    )
    parser.add_argument(
        "--speakers", type=_speakers_type, default="1",
        help='Diarize and assign a voice per speaker: "1" (default, off), "auto" (auto-detect count), '
             'or a fixed count like "3"'
    )
    parser.add_argument(
        "--voice-map", type=_parse_voice_map, default=None,
        help='Per-speaker voice overrides, e.g. "SPEAKER_00=Phạm Tuyên,SPEAKER_01=Ngọc Huyền"'
    )

    args = parser.parse_args()

    if args.install_skill:
        _install_skill()
        sys.exit(0)

    pipeline_config.load(args.config)

    if args.style == "list":
        _print_styles()
        sys.exit(0)

    if not args.input or not args.output or not args.language:
        parser.error("input, output, and --language are required (unless using --style list or --install-skill)")

    from pipeline.llm_client import validate_env
    missing = validate_env(pipeline_config.get())
    if missing:
        sys.stderr.write(
            f"ERROR: missing required environment variable(s): {', '.join(missing)}\n"
            f"       Set them in .env or export them before running.\n"
        )
        sys.exit(1)

    if args.no_voiceover:
        voiceover = False
    elif args.voiceover:
        voiceover = True
    else:
        voiceover = True

    style_name = args.style or pipeline_config.get()["preferences"].get("style", "standard")
    style = resolve_style(style_name)
    subtitle_formats = tuple(
        fmt.strip().lower()
        for fmt in args.subtitle_formats.split(",")
        if fmt.strip()
    ) or ("srt",)

    prefer_source_captions = (
        pipeline_config.get()["transcription"].get("prefer_source_captions", True)
        and not args.no_source_captions
    )

    translate_video(
        args.input,
        args.output,
        args.language,
        args.voice,
        not args.no_subtitles,
        args.source_language,
        style,
        voiceover,
        subtitle_formats=subtitle_formats,
        burn_subtitles=args.burn_subtitles,
        timings_out=args.timings_out,
        prefer_source_captions=prefer_source_captions,
        fit=False if args.no_fit else None,
        subtitle_lang=args.subtitle_lang,
        speakers=args.speakers,
        voice_map=args.voice_map,
    )


if __name__ == "__main__":
    main()
