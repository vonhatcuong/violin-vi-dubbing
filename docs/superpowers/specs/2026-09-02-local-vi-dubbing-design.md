# Thiết kế: lồng tiếng video EN → VI hoàn toàn local trên violin

- Ngày: 2026-09-02
- Trạng thái: đã duyệt (plan mode) — nguồn: `~/.claude/plans/t-i-mu-n-x-y-d-ng-shimmering-bengio.md`
- Branch: `feat/local-vi-dubbing`
- Phạm vi: mở rộng fork violin tại chỗ; mọi tính năng mới tắt mặc định trong `config/default.yaml`

## Context

Mục tiêu: dựng pipeline lồng tiếng video tiếng Anh sang tiếng Việt, giữ nghĩa, khớp mốc thời gian từng câu với video gốc, giữ nhạc nền, chạy hoàn toàn local (không API trả phí, không dịch vụ cloud).

Quyết định của user (đã hỏi):
- Phần cứng: dev trên Mac M3 Pro 18 GB, batch trên GPU NVIDIA (DGX 8×H100 qua SLURM, hoặc box thuê).
- Video nguồn: bài giảng/tutorial 1 người nói + phỏng vấn/podcast nhiều người.
- Giọng: giọng Việt cố định chất lượng cao (không clone giọng gốc).
- Đồng bộ: khớp thời gian từng câu, giữ nhạc nền; không cần lip-sync.

## Hiện trạng trong workspace (đã khảo sát)

`violin/` là fork của shang-zhu/violin (MIT), đã tuỳ biến local, có preset `config/local_mac.yaml`, đã chạy ra `output/out_vi.mp4`. Kiến trúc 5 bước trong `pipeline/orchestrator.py:dub_video`:
extract audio → `transcribe` (faster-whisper `large-v3-turbo` int8 CPU, `pipeline/transcriber_local.py`) → `merge_continuous_segments` → `translate_segments` (LLM qua OpenAI-compatible client, `pipeline/llm_client.py`) → re-merge/`split_into_sentences` → `synthesize_segments` (`pipeline/tts.py` chọn backend) → `build_aligned_video` (`pipeline/merger.py`: stretch video ≤ 8%, atempo ≤ 1.4×, gap chunks giữ audio gốc) → SRT/VTT/TXT.

Đã có sẵn và tái dụng được:
- `prompts/translate.yaml`: prompt dịch length-aware (mục tiêu ~0.85× số từ, cấm code-switching, có `asr_corrections_block`).
- `two_pass_tts.py`: synth 1.0× → đo → re-synth với speed riêng cho segment tràn slot (Supertonic).
- `pipeline/tts_supertonic.py`: TTS ONNX local, 31 ngôn ngữ gồm `vi`, 10 giọng M1–M5/F1–F5.
- `Segment` dataclass có sẵn trường `speaker` (`pipeline/transcriber.py:81`) nhưng chưa dùng.
- Tests (12 file) + FastAPI + SQLite job history.

Chưa local / còn thiếu so với yêu cầu:
1. Dịch: `local_mac.yaml` trỏ `provider: ollama` + `OLLAMA_API_KEY` = Ollama **Cloud**. Ollama chưa cài trên máy.
2. TTS: `provider: edge` = Edge-TTS (dịch vụ Microsoft qua mạng). Supertonic local có nhưng chất lượng tiếng Việt chưa đánh giá.
3. Không diarization → podcast nhiều người bị 1 giọng.
4. Không tách vocal → nhạc nền/sfx bị hạ xuống 2% cùng giọng gốc (`voiceover_volume: 0.02`), không giữ được nhạc nền sạch.
5. Vòng lặp fit-duration mới ở dạng script rời (`two_pass_tts.py`), chưa nằm trong orchestrator; chưa có bước "dịch lại ngắn hơn" khi TTS vẫn tràn.

## Nghiên cứu open-source (2026-09)

Pipeline end-to-end đã khảo sát: pyvideotrans (18.9k★, GPL-3), VideoLingo (18.3k★, Apache-2), open-dubbing (Softcatala, Apache-2), SoniTranslate, Linly-Dubbing (dormant), KrillinAI (Go, TTS cloud-only), Voxa (MIT, single-file, slot-anchored). Không dự án nào ship TTS tiếng Việt local mặc định; SeamlessM4T không có speech output tiếng Việt.

Bài học thiết kế rút ra (áp dụng vào plan):
1. **Anchor theo slot, không concat**: mỗi câu chiếm slot [onset → onset câu kế]; tràn thì cắt/tăng tốc, thiếu thì đệm im lặng. Concat tích luỹ drift ~18 s/giờ.
2. **Mượn khoảng lặng phía sau trước khi tăng tốc** (open-dubbing): ratio = dur_TTS / (onset_kế − start), không phải / (end − start).
3. **Rút gọn text trước khi ép audio** (VideoLingo): ước lượng duration từ số âm tiết trước TTS → nếu tràn, prompt LLM rút gọn; atempo ≤ 1.2 chấp nhận, ≤ 1.4 cứng; không bao giờ làm chậm giọng.
4. **Tiếng Việt đơn âm tiết**: số từ ≈ số âm tiết, nhưng hằng số giây/âm tiết phải đo thực nghiệm theo từng giọng TTS.
5. **Merge rồi mới split** theo nghĩa (spaCy/LLM), tránh câu bị cắt ngang giữa 2 segment ASR.
6. **Timestamp ảo/trôi**: align word-level trên vocal stem, sàn duration 2.5 s, merge segment dưới sàn; có thể re-ASR audio đã lồng để dựng lại SRT trung thực (pyvideotrans).
7. **Tách trước, mix sau**: Demucs `htdemucs` (chạy MPS) → lồng trên vocal stem, overlay lại nhạc nền, vocal lồng cao hơn nền ~5 dB.
8. **Nhiều người nói**: pyannote diarization → gán giọng theo speaker; fallback rẻ là phân loại giới tính → giọng nam/nữ preset.
9. **Quality gate TTS**: ASR ngược lại clip TTS, chấm WER/clipping/silence, re-synth tối đa 2 lần (Voxa).
10. **Mac**: WhisperX không có MPS; faster-whisper CPU / whisper.cpp / mlx-whisper ổn hơn.

Tài liệu kỹ thuật: Virkar 2022 (prosodic alignment), Lakew (isochrony-aware NMT, Amazon), VideoDubber (AAAI 2023), IWSLT 2023 dubbing track.

## Stack đã chốt (đã xác minh 2026-09-02)

| Stage | Mac M3 Pro 18 GB (dev) | GPU NVIDIA (batch) | Ghi chú |
|---|---|---|---|
| Tách vocal | `demucs-mlx` 1.4.6 (htdemucs trên MLX; PyTorch-MPS của Demucs không ổn định) | `demucs` 4.1.0 cuda | Giữ nhạc nền; ASR chạy trên `vocals.wav` |
| ASR EN | faster-whisper `large-v3-turbo` int8 CPU (đã có `pipeline/transcriber_local.py`) | cùng adapter, `local_device: cuda` (WhisperX là tuỳ chọn sau) | Word timestamps đã có |
| Diarization | pyannote.audio 4.0.7, model `pyannote/speaker-diarization-community-1` (CC-BY-4.0, gated tự động, HF token free, chạy offline sau khi tải) | cuda | Chỉ bật khi `--speakers auto` |
| Dịch EN→VI | Ollama local `qwen3.5:9b-mlx` (≈5.5 GB) hoặc `gemma4:12b-mlx` (≈7.5 GB) | vLLM/Ollama `qwen3.5:27b` qua `base_url` OpenAI-compatible | Cùng một client OpenAI SDK, đổi `base_url` |
| Chuẩn hoá text VI | `vinorm` 2.0.7 (số, ngày, viết tắt) + lowercase | như Mac | Model F5-vi train trên text lowercase |
| TTS VI | **F5-TTS chính thức (PyPI `f5-tts` 1.1.22, PyTorch MPS)** + checkpoint `hynt/F5-TTS-Vietnamese-ViVoice` (`model_last.pt`, `config.json` → rename `vocab.txt`), `--model F5TTS_Base`, vocoder vocos | cùng code, cuda | CC-BY-NC-SA (cá nhân/nghiên cứu OK). Có `speed` và `fix_duration`. Không dùng `f5-tts-mlx` (ngừng cập nhật 03/2025, convert ckpt tuỳ biến rủi ro) |
| Giọng cố định | Voice bank: clip mẫu 5–10 s tiếng Việt + ref text, `assets/voices/vi/` | như Mac | Nam/Nữ, Bắc/Nam; user tự cung cấp clip có bản quyền |
| Merge | ffmpeg (đã có `pipeline/merger.py`) | như Mac | Video stretch ≤ 8 % đã hỗ trợ |

Loại bỏ: Edge-TTS (mạng), Ollama Cloud, Supertonic (giữ code nhưng không mặc định), Together/ElevenLabs/OpenAI (giữ nguyên, không đụng).

## Thiết kế: mở rộng violin tại chỗ

Nguyên tắc: giữ nguyên cấu trúc 5 bước và pattern backend-theo-config của violin; chèn 3 stage mới (separate, diarize, fit) và 1 TTS backend mới; mọi stage ghi artifact JSON cạnh output để resume/sửa tay. Mọi tính năng mới tắt trong `config/default.yaml` để 12 test cũ và đường cloud không đổi.

### Lỗi có sẵn phải sửa trước (M1)
- `pipeline/translator.py:298` dựng lại `Segment` không có `speaker` → mất thông tin diarization. Sửa: truyền `speaker=s.speaker, source_text=s.text`.
- `merge.max_subtitle_chars: 20` (`config/default.yaml:58`) khiến `split_into_sentences` băm câu dịch thành đơn vị < 1 s → TTS rời rạc. Trong `local_mac.yaml` đặt `0`; thêm `merge.min_duration: 2.5` vào `merge_continuous_segments` để gộp đơn vị quá ngắn.

### Data model
- `Segment` thêm `source_text: str = ""` (`pipeline/transcriber.py:81`); `_persist_segments` dùng `asdict` nên tự ghi ra JSON, người dùng sửa bản dịch cạnh câu gốc rồi resume.
- Trạng thái fit nằm trong dataclass `DubUnit` mới ở `pipeline/fitter.py` (`seg_id, speaker, voice, source_text, text, start, slot_end, budget_s, syllables, est_s, tts_path, tts_dur, tts_speed, atempo, strategy, rounds`), ghi `<output>.fit.units.json`. Không nhét vào `Segment` vì subtitles/API/tests đều dùng `Segment`.
- Khi `fit.enabled`: tách câu theo word-timestamp **trước** khi dịch (đã có `_split_words_into_sentences`, `transcriber.py:221`), dịch 1:1 theo batch (LLM vẫn thấy ngữ cảnh 16 câu/batch), **bỏ** bước re-merge + `split_into_sentences` sau dịch (`orchestrator.py:140-141`). Onset câu lấy từ ASR, không chia theo tỉ lệ ký tự.

### Luồng mới trong `pipeline/orchestrator.py:dub_video`

```
extract_audio (16k, ASR)  ──┐
separate (Demucs) → vocals.wav / no_vocals.wav (44.1k)   [M2, tuỳ chọn]
transcribe(vocals_16k)                                    [có sẵn]
diarize(vocals) → gán seg.speaker                         [M3, tuỳ chọn]
merge_continuous_segments → translate_segments (kèm budget giây/âm tiết)
re-merge → split_into_sentences                           [có sẵn]
fit_segments (fitter.py): ước lượng → rút gọn (LLM) → TTS pass 1 → đo → pass 2 speed
prepare_merge(bed=no_vocals) + build_gap_chunks ‖ (TTS đã xong trong fit)
build_aligned_video → subtitles                           [có sẵn, sửa mix]
```

### Module mới

0. `pipeline/devices.py` — `pick_device(requested) -> "mps"|"cuda"|"cpu"`, `free_memory()` (gc + `torch.mps/cuda.empty_cache`) gọi sau mỗi stage nặng.
1. `pipeline/separator.py` — `separate(audio_path, out_dir) -> Stems(vocals, no_vocals)`; backend `demucs_mlx` (Mac) hoặc `demucs` (cuda) theo `separation.backend`; cache theo hash file; `unload()`; khi `separation.enabled: false` trả về `Stems(vocals=audio_gốc, no_vocals=None)`.
2. `pipeline/diarizer.py` — `diarize(vocals_path, min_speakers, max_speakers) -> list[Turn(start,end,speaker)]`; `assign_speakers(segments, turns)` gán speaker theo overlap lớn nhất; pure-python phần assign (test được không cần model).
3. `pipeline/vi_text.py` — `normalize_for_tts(text) -> str`: `vinorm` → lowercase → thay ký tự ngoài vocab; `count_syllables(text) -> int` (đếm token cách nhau bằng khoảng trắng sau normalize; tiếng Việt đơn âm tiết).
4. `pipeline/tts_f5vi.py` — backend theo contract của `pipeline/tts.py` (giống `tts_supertonic.py`): `get_shared_tts()` load 1 lần (`load_model(DiT, F5TTS_Base cfg, ckpt_file, vocab_file)` từ `f5_tts.infer.utils_infer`), `synthesize_segment(text, voice, out, client, language, speed, emotion)` gọi `infer_process(ref_audio, ref_text, gen_text, model, vocoder, speed=..., fix_duration=None)`, ghi WAV 44.1k mono (resample từ 24k bằng ffmpeg như `_write_wav_44k_mono`), pad tail silence như Supertonic; `native_voices_for("vi")` đọc voice bank; `synthesize_segments(..., speeds: list[float] | None)` nhận speed theo từng segment.
5. `pipeline/voices.py` — đọc `assets/voices/vi/catalog.yaml` (`name, gender, wav, ref_text, description`); `assign_voices(speakers, default_voice, voice_map, genders) -> dict[speaker, voice]`; `guess_gender(vocals_path, turns)` theo F0 trung vị (librosa `pyin`, ngưỡng ~165 Hz) [M3].
6. `pipeline/fitter.py` — trái tim đồng bộ:
   - `compute_slots(segments, total_duration, max_borrow_s, margin_s)`: slot_i = min(end_i + max_borrow, start_{i+1} − margin) − start_i.
   - `estimate_duration(text_vi, sec_per_syllable)`.
   - `fit_segments(segments, slots, llm_client, tts_synth, cfg) -> FitResult(tts_paths, tts_durs, segments_out, log)`:
     1. Nếu est > slot × `fit.overrun_tolerance` (1.30) → `translator.shorten_segment(src, vi, budget_syllables)` tối đa `fit.max_shorten_rounds` (2).
     2. TTS pass 1 speed 1.0 cho tất cả; đo duration.
     3. Segment tràn: speed = min(dur/slot, `fit.tts_speed_max` 1.15) → re-synth pass 2 (chỉ segment tràn).
     4. Ghi `seg.end = start + min(dur_TTS, slot)` khi mượn pause (kéo dài slot cho merger, chảy tự nhiên vào gap math `merger.py:199-211`); ghi `unit.atempo` ≤ `fit.max_atempo` (1.25) cho phần dư; phần còn lại giao merger (video ≤ 8 % → atempo tới 1.4 → hard trim có fade).
     5. Ghi `<output>.fit.units.json`.
   - Tách 2 pha: `fit_text` (chỉ cần LLM) rồi `fit_audio` (chỉ cần TTS) để LLM và F5 không phải cùng nằm trong RAM.
   - Không bao giờ làm chậm giọng; segment ngắn hơn slot giữ nguyên (merger đệm im lặng bằng `apad` sẵn có, `speed_clamp_max: 1.0`).
   - `calibrate_voice(voice)`: synth 3 câu mẫu, đo sec/âm tiết, cache vào `~/.cache/violin/calib.json`; override `fit.sec_per_syllable`.

### Sửa file có sẵn

- `pipeline/llm_client.py`: thêm `models.translation.base_url` + `api_key` tuỳ chọn; provider `ollama` mặc định `http://localhost:11434/v1`, key mặc định `"ollama"` (không bắt buộc env). Cập nhật `required_env_keys`/`validate_env` để không đòi `OLLAMA_API_KEY` khi base_url là localhost. `/no_think` đã có trong `translator.py:121,190`.
- `pipeline/translator.py`: sửa dòng 298 (giữ `speaker`, `source_text`); thêm `shorten_segment(source_text, current_vi, budget_syllables, budget_seconds, client) -> str` dùng prompt mới `shorten_system/shorten_user` (JSON `{"translation": ...}` như `SINGLE_SCHEMA`). `_try_batch`: numbered_segments kèm `(x.x s, ≤ N âm tiết)` khi có slot. Thêm `translation.response_format: json_schema | json_object` (Ollama từng treo với `json_schema`, xem commit 3957df5).
- `prompts/translate.yaml`: thêm budget vào `batch_user` (thay quy tắc 0.85× mơ hồ bằng ngân sách âm tiết cụ thể theo slot), thêm `shorten_*`.
- `pipeline/tts.py`: thêm nhánh `f5vi` trong `_backend` và `_make_client`; `synthesize_segments` nhận `speeds` tuỳ chọn (backend không hỗ trợ thì bỏ qua).
- `pipeline/merger.py`: `MergePlan` thêm `bed_audio_path` (no_vocals); trong `_build_video_audio_track`, entry `tts_mixed` trộn bed (speed-adjusted cùng video) ở `bed_volume` (1.0) + `dub_gain_db` (+5 dB cho giọng lồng) thay vì audio gốc ở `voiceover_volume`; gap giữ audio gốc đầy đủ. `build_aligned_video` nhận `applied_atempo: list[float]` để trần `max_audio_speedup` trừ đi phần fitter đã áp (tránh tăng tốc 2 lần, `merger.py:342`). Thêm `merge_video.hard_trim` (cắt tại `target_dur` + afade 80 ms) thay cho freeze còn sót (`merger.py:352-355`). Không có stems → hành vi cũ.
- `pipeline/orchestrator.py`: chèn separate/diarize/fit như sơ đồ; `DubOptions` thêm `speakers: str = "1"` (`"1"|"auto"|"N"`), `voice_map: dict[str,str] | None`, `separation: bool | None`. Chuyển `prepare_merge`/gap thread (dòng 151-168) xuống **sau** fit vì `seg.end` có thể đổi. Persist thêm `diarized`, `fitted` segments + `voices.json` + `fit.units.json`. `resume_from_segments.py` nhận stage `fitted`.
- `main.py`: flags `--speakers`, `--voice-map SPEAKER_00=nam-1,SPEAKER_01=nu-1`, `--no-separation`, `--voice` liệt kê voice bank khi `--voice list`.
- `config/default.yaml`: khối mới `separation`, `diarization`, `fit`, `f5vi`, `voices` với mặc định tắt/an toàn để test cũ không đổi.
- `config/local_mac.yaml`: translation `provider: ollama`, `model: qwen3.5:9b-mlx`, `base_url: http://localhost:11434/v1`; tts `provider: f5vi`; bật separation, diarization auto; `fit` bật.
- `config/local_gpu.yaml` (mới): như trên nhưng `local_device: cuda`, `f5vi.device: cuda`, `qwen3.5:27b`, workers cao hơn.
- `pyproject.toml`: extra `local-mac` (faster-whisper, f5-tts, vinorm, demucs-mlx, pyannote.audio, librosa) và `local-gpu` (faster-whisper, f5-tts, vinorm, demucs, pyannote.audio, librosa); giữ extra `local` cũ làm alias của `local-mac`.
- `two_pass_tts.py`: giữ lại nhưng ghi chú deprecated, logic đã chuyển vào `fitter.py`.

### Bộ nhớ trên Mac 18 GB
Chạy tuần tự, giải phóng model giữa các stage: Demucs (~2 GB) → unload; faster-whisper (~2 GB) → unload; Ollama server giữ LLM (~5.5–7.5 GB) chỉ trong translate + shorten (shorten thực hiện trước TTS, sau đó gọi `keep_alive: 0` để unload); F5-TTS (~2 GB) + pyannote (~1 GB) cuối. Không stage nào cần > 10 GB đồng thời.

### Kiểm thử
- Unit (không cần model): `tests/test_fitter.py` (slot/borrow/estimate/quyết định speed với fake TTS trả WAV im lặng dài = syllables × k; fake LLM trả JSON rút gọn), `tests/test_vi_text.py` (normalize số/viết tắt, đếm âm tiết), `tests/test_diarizer_assign.py` (gán speaker theo overlap), `tests/test_voices.py` (assign theo gender/map), `tests/test_llm_client_local.py` (ollama local không cần key), `tests/test_merger_bed.py` (mix bed bằng ffmpeg trên WAV sine 3 s, kiểm tra RMS).
- Tests cũ (12 file) phải pass nguyên trạng — default config tắt tính năng mới.
- Integration: clip 10 s tạo bằng ffmpeg (`color` + `sine`) chạy `dub_video` với backend fake → ra mp4 + srt, độ dài video ≤ 1.08× gốc, mỗi segment bắt đầu đúng onset ± 50 ms.
- E2E thủ công: `uv run main.py <video 2–3 phút> out_vi.mp4 --language Vietnamese --config config/local_mac.yaml --speakers auto`; nghe kiểm tra, xem `out_vi.fit.json` (tỉ lệ segment cần speed > 1.15 mục tiêu < 10 %).

### Milestones (mỗi mốc verify độc lập)
- **M0 Spikes (trước khi code, ghi kết quả vào `docs/superpowers/specs/…-spikes.md`)**: (a) F5-TTS-vi chạy MPS với ckpt hynt (`f5-tts_infer-cli --model F5TTS_Base --vocab_file vocab.txt --ckpt_file model_last.pt`), đo RTF và sec/âm tiết ở speed 1.0 và độ tự nhiên ở speed 1.15 → điền `fit.sec_per_syllable`, `fit.max_tts_speed`; (b) `brew install ollama` + `ollama pull qwen3.5:9b-mlx`, chạy prompt batch của violin với `json_object` + `/no_think`, kiểm tra ổn định số lượng ở batch 16; (c) `demucs-mlx` trên clip 3 phút: RTF, RAM; (d) pyannote `community-1` tải được với HF token, chạy offline, cpu vs mps; (e) `vinorm` cài được trên arm64 (fallback: `num2words` 0.5.14 `lang="vi"` + regex).
- **M1 Local single-speaker**: sửa 2 lỗi có sẵn + llm_client local + tts_f5vi + vi_text + fitter + prompt budget + config local_mac. Verify: E2E trên `output/` sample, không có kết nối mạng (tắt Wi-Fi vẫn chạy); `fit.units.json` không có unit nào vượt atempo 1.4.
- **M2 Stems**: separator + merger bed mixing. Verify: nhạc nền nghe rõ ở đoạn có lời; test_merger_bed.
- **M3 Multi-speaker**: diarizer + voices + gender heuristic + CLI flags. Verify: podcast 2 người ra 2 giọng, `*.diarized.segments.json` đúng.
- **M4 GPU**: `config/local_gpu.yaml` + script SLURM/Singularity cho DGX (`dgx-access` docs) hoặc box 4090; verify cùng output với Mac, nhanh hơn.
- **M5 (tuỳ chọn)**: quality gate re-ASR WER trên clip TTS, re-synth ≤ 2 lần; WhisperX adapter cho GPU.

### Rủi ro chính
1. F5-vi trên MPS chậm (1–7× RTF): dev trên clip ngắn, batch trên GPU. 2. `fix_duration`/speed > 1.15 làm giọng méo → dựa vào rút gọn dịch là chính. 3. Số/viết tắt/từ mượn tiếng Anh làm sai dấu → `vi_text` + từ điển loanword tuỳ chỉnh trong config. 4. License CC-BY-NC-SA của F5-vi: chỉ dùng cá nhân/nghiên cứu (user đã chọn). 5. pyannote cần HF token 1 lần (vẫn chạy local).

## Amendment 2026-09-02 (quyết định của user sau khi có server GPU)

- **LLM dịch**: Google **Gemma 4** thay cho Qwen 3.5 — Mac: `gemma4:12b-mlx` (Ollama MLX, ~7,5 GB), GPU 24 GB: `gemma4:31b` (Q4, ~19 GB). Cùng client OpenAI-compatible, chỉ đổi `models.translation.model`.
- **TTS**: **VieNeu-TTS v3 Turbo** (`pnnbao-ump/VieNeu-TTS-v3-Turbo`, Apache-2.0, 48 kHz, 20 giọng preset Bắc/Trung/Nam, clone giọng từ clip 3–8 s, code-switching Anh–Việt) thay cho F5-TTS-Vietnamese. SDK `vieneu` 3.3.0: trên CPU/Apple Silicon chạy ONNX int8 không cần torch (~7× realtime), trên CUDA chạy PyTorch batch (`torch==2.8.0` cu128 + `transformers==4.57.6`). `infer()` không có tham số tốc độ → fitter chỉ đo và ghi `over_s`; merger (video ≤ 8 %, atempo ≤ 1.4, hard trim) hấp thụ phần dư. Giọng cố định mặc định: `Phạm Tuyên` (nam, Bắc) / `Ngọc Huyền` (nữ, Bắc); voice bank chỉ còn dùng cho clone tuỳ chọn. Hệ quả: license sạch cho thương mại, Mac chạy được TTS nhanh.
- **`vinorm` bị loại** (import module `imp` đã xoá ở Python 3.12); chuẩn hoá text dùng `pipeline/vi_text.py` (num2words + quy tắc nghìn/thập phân/âm).
- **Hạ tầng**: spike và E2E chạy trên server Vast.ai 2× RTX 3090 (Ollama trên GPU 1, ASR/TTS trên GPU 0); Mac giữ nhẹ theo yêu cầu user (chỉ chạy test unit).
