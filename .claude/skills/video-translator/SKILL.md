---
name: video-translator
description: Dub a video URL, video file, or audio file into another language and generate subtitles/transcript. Trigger when the user wants to translate / dub / voice-over a YouTube URL or local media file, or generate subtitles for it. Handles common video files plus `.mp3` / `.wav` / `.m4a` / `.flac`. Installs as the `violin` CLI (and `violin-api` for the FastAPI server) via `uv tool install`. Use `config/local_mac.yaml` for the local Vietnamese workflow with faster-whisper + Edge-TTS.
allowed-tools: Bash, Read
---

# Violin — operating skill

Default CLI uses the packaged cloud config. For the local Vietnamese workflow, run from the repo with `--config config/local_mac.yaml`; this keeps media processing, transcription, TTS, timestamp alignment, subtitle generation, and burn-in local. Translation uses the configured LLM provider.

## Pre-flight

Run these silently first. Abort if the CLI/input check fails, or if the key for the selected config is missing:

```bash
command -v violin                 # 1. CLI on PATH
test -f "<input>"                 # 2. Input exists
printenv TOGETHER_API_KEY         # 3a. Default config key
printenv OLLAMA_API_KEY           # 3b. local_mac.yaml key
```

If `violin` is missing: tell the user to `uv tool install violin`, then `violin --install-skill` to refresh this skill file. Do not auto-install.

For URL inputs, skip the `test -f` check. Only require the key for the config being used. If using `config/local_mac.yaml`, check `OLLAMA_API_KEY` instead of `TOGETHER_API_KEY`. Edge-TTS itself does not need a key.

## Decisions

- **CLI vs API**: single run-and-wait file → CLI (`violin`). Multi-job / HTTP / web UI → API server (`violin-api`); print the command, don't auto-start it.
- **Vietnamese local default**: if the user says "Vietnamese", "tiếng Việt", or asks for local Edge-TTS, use `--language Vietnamese --config config/local_mac.yaml --subtitle-formats srt,vtt,txt --burn-subtitles`.
- **Fully offline Vietnamese**: `--config config/local_mac.yaml` also works fully offline (no API keys) — it runs transcription, translation, and TTS entirely on-device.
- **Style** (`--style`): default `standard`. Kids content → `kids`, formal/lecture → `academic`, casual → `casual`, dramatic → `storyteller`, news → `news`. Run `violin --style list` if unsure.
- **Voiceover**: keep default (mix dubbed audio over a quiet original). Use `--no-voiceover` only when the user explicitly says "replace audio entirely".

## Run

```bash
violin <input-file-or-url> <output> --language <Lang> [flags]
```

## Flags

| Flag | Default | When to set |
|------|---------|-------------|
| `--language` / `-l` | *required* | Target language (e.g. `Chinese`, `Spanish`, `Japanese`). |
| `--voice` / `-v` | auto (native voice picked by `preferences.voice_gender`) | Only when the user names a specific voice from the catalog (e.g. `"warm female narrator"`). Otherwise omit and let the default kick in. |
| `--source-language` | `auto-detect` | Only if Whisper mis-detects the source language. |
| `--style` / `-s` | `standard` | See Decisions above. |
| `--subtitle-formats` | `srt` | Use `srt,vtt,txt` when the user wants all subtitle/transcript-style sidecars. |
| `--burn-subtitles` | off | User wants subtitles burned into a second MP4. |
| `--no-subtitles` | off | User says "no subtitles" / "video only". |
| `--no-voiceover` | off | User says "replace original audio entirely". |
| `--config` / `-c` | `config/default.yaml` | Use `config/local_mac.yaml` for local faster-whisper + Edge-TTS Vietnamese jobs. |
| `--timings-out` | off | Only when the user wants a per-step timing JSON for debugging / benchmarking. |

## Language coverage

33 target languages total. **16** ship with handpicked native-speaker voices: Chinese, Spanish, English, Hindi, Arabic, Portuguese, Russian, Japanese, Turkish, German, Korean, French, Italian, Polish, Dutch, Swedish. The other **17** fall back to the English voice catalog (multilingual under Cartesia Sonic 3) — quality is decent but the voice isn't a native speaker. Mention this caveat only if the user is translating to a fallback language and asks about voice quality.

## Report back

- Output video path, subtitle paths, burned-subtitle video path, and transcript path (printed by the run).
- Total cost (printed at end — surface, don't hide).
- If voiceover was on, mention the `_original.m4a` sidecar.

## Don'ts

- Don't run on multi-GB videos without first quoting the rough cost (audio length × per-provider rates in `pipeline/pricing.py`).
- Don't fabricate a "subtitles-only" mode — the CLI requires the full pipeline. If the user only wants SRT, run the full pipeline and hand them just the `.srt`, warning them of the cost first.
- Don't claim the workflow is fully offline when translation config points to a remote LLM provider.
- Don't paraphrase the README. For supported languages (33), voice catalog, and full flag docs, point them at `README.md` or `violin --help`.
