# Thiết kế: Ưu tiên transcript/caption YouTube làm nguồn script

- **Ngày:** 2026-05-29
- **Trạng thái:** Đã duyệt thiết kế — chờ viết implementation plan
- **Phạm vi:** `violin` (CLI + FastAPI web app)

---

## 1. Bối cảnh & vấn đề

Hiện tại pipeline **luôn** chạy Whisper (`pipeline/orchestrator.py:113` → `transcribe()`) trên audio đã extract, bất kể nguồn là YouTube hay file. `pipeline/downloader.py` chỉ tải media bằng yt-dlp, **không** lấy caption (`writesubtitles`/`writeautomaticsub` đều không có). Cả CLI (`main.py`) lẫn API (`api/worker.py`) đều theo luồng `download → transcribe`.

Hệ quả: mọi video YouTube đều bị transcribe lại từ đầu — chậm hơn, tốn STT hơn, và bỏ phí script gốc (manual caption của creator thường chính xác hơn cho tên riêng/thuật ngữ).

**Mục tiêu:** Khi input là URL có caption, ưu tiên dùng caption đó làm nguồn script (rồi mới dịch + TTS + căn timestamp như cũ), chỉ fallback Whisper khi không có caption khả dụng.

## 2. Mục tiêu & phi mục tiêu

**Mục tiêu**
- Lấy caption YouTube qua yt-dlp, chuẩn hoá thành `list[Segment]` đưa vào pipeline hiện có.
- Ưu tiên: manual caption → auto-caption (ASR gốc) → Whisper.
- Auto-caption (thiếu dấu câu) được LLM khôi phục dấu câu rồi **re-align về timestamp word-level** (giữ độ chính xác mốc giờ).
- Bật mặc định, có cờ tắt để ép Whisper.
- **Không thay đổi** cơ chế căn chỉnh trong `merger.py`.

**Phi mục tiêu (v1)**
- Không tối ưu riêng cho site ngoài YouTube (nếu yt-dlp lấy được caption thì tự hưởng, không đảm bảo).
- Không dùng track **dịch tự động** của YouTube (luôn lấy caption *ngôn ngữ nguồn* rồi để pipeline tự dịch).
- Không cache caption.
- Không thay đổi thuật toán align video↔TTS.

## 3. Quyết định đã chốt (qua brainstorming)

| # | Quyết định | Lựa chọn |
|---|-----------|----------|
| 1 | Mức tận dụng caption | **Manual → Auto (LLM thêm dấu câu) → Whisper** |
| 2 | Mặc định & control | **Bật mặc định + flag tắt** (`--no-source-captions`, config, checkbox web) |
| 3 | Xử lý auto-caption | **Word-level re-align** (chính xác nhất), có fallback đơn giản |

## 4. Kiến trúc & luồng dữ liệu

```
URL ──► yt-dlp tải video (như cũ — vẫn cần video để mux)
    └─► captions.fetch_source_captions(url, source_language)
            ├─ extract_info(download=False) → _select_track
            ├─ _download_json3(url) → parse:
            │     • manual → _parse_manual         → Segment[] (đã có dấu câu)
            │     • auto   → _parse_auto_words      → words[]
            │                 → _restore_punctuation_and_align → Segment[]
            └─ trả Segment[]  |  None
    ├─ Segment[] → dub_video(..., segments_override=segs)   # BỎ QUA extract_audio + Whisper
    └─ None      → dub_video(...)                            # Whisper (fallback, như cũ)
```

`dub_video` với `segments_override`: bỏ `extract_audio` + `transcribe`; vẫn lấy `total_duration = get_video_duration()` + `ensure_video_input()`; rồi chạy `merge_continuous_segments → _persist_segments("transcribed") → translate → ... → build_aligned_video` y như hiện tại.

## 5. Module mới `pipeline/captions.py`

Một file, một trách nhiệm: lấy & chuẩn hoá caption thành `Segment[]`. **Không** import/sửa `merger.py`; chỉ import `Segment` từ `transcriber.py`.

```python
def fetch_source_captions(
    url: str,
    source_language: str = "auto-detect",
    *,
    prefer_manual: bool = True,
    llm_client=None,           # dùng cho restore punctuation; None → lấy make_translation_client
    tracker: CostTracker | None = None,
) -> list[Segment] | None:
    """Trả Segment[] nếu lấy được caption ngôn ngữ nguồn; None nếu không (caller fallback Whisper)."""
```

Helpers nội bộ:
- `_select_track(info, src_code) -> TrackRef | None`
- `_download_json3(url) -> dict`
- `_parse_manual(events) -> list[Segment]`
- `_parse_auto_words(events) -> list[_Word]`  (`_Word = (text, start, end)`)
- `_restore_punctuation_and_align(words, llm_client, source_language, tracker) -> list[Segment]`

## 6. Chọn track caption (`_select_track`)

Đây là điểm dễ sai nhất (`automatic_captions` chứa cả bản dịch tự động sang mọi ngôn ngữ, vd `aa-en`).

1. **Mã ngôn ngữ nguồn**: nếu `source_language != "auto-detect"` → `languages.language_code(source_language)`; ngược lại → `info.get("language")` (yt-dlp metadata). Nếu vẫn None → bước 2 dùng track manual đầu tiên.
2. **Thứ tự ưu tiên**:
   1. `subtitles[src]` (manual, đúng ngôn ngữ)
   2. `subtitles[k]` với `k` bắt đầu bằng `src` (vd `en-US`)
   3. `automatic_captions[src]` (ASR gốc — key trùng đúng mã ngôn ngữ nguồn)
   4. nếu chưa biết `src`: track **manual** đầu tiên có trong `subtitles`
3. **Loại bản dịch tự động**: khi lấy từ `automatic_captions`, **chỉ** nhận key trùng đúng mã ngôn ngữ nguồn (ASR gốc); bỏ mọi key dạng `<x>-<y>` không khớp.
4. Trong track đã chọn, lấy entry có `ext == "json3"`. (json3 có word-level cho auto và cấu trúc cue rõ ràng cho manual.)
5. Trả `kind ∈ {manual, auto}`, `lang`, `url`.
6. **Trường hợp an toàn**: nếu không xác định được mã ngôn ngữ nguồn **và** không có manual caption nào → trả `None` (fallback Whisper), thay vì đoán mò một track auto có thể là bản dịch.

## 7. Parse manual (`_parse_manual`) — cue-level, đã có dấu câu

- Mỗi `event` có `segs`: `text = "".join(seg["utf8"])`, `start = tStartMs/1000`, `end = (tStartMs + dDurationMs)/1000`.
- Bỏ event không có `segs` hoặc `text.strip() == ""`.
- → `Segment(id, start, end, text)`.
- `merge_continuous_segments` + `split_into_sentences` (đã có trong orchestrator) tự gộp cue thành câu hoàn chỉnh rồi cắt lại — không cần xử lý thêm.

## 8. Parse auto + restore + re-align

### 8a. `_parse_auto_words` — gom word-level + dedup phòng thủ
- Với mỗi `event` có `segs`, mỗi `seg` có `utf8` không rỗng và khác `"\n"`: `abs_start = (event.tStartMs + seg.tOffsetMs)/1000` (tOffsetMs thiếu ⇒ 0).
- Sort theo `abs_start`.
- **Dedup rolling-overlap**: bỏ một từ nếu trùng `text` (casefold) **và** `|abs_start − abs_start_của_từ_đã_giữ_gần_nhất| < 0.10s`.
- `end[i] = start[i+1]` nếu có; từ cuối `end = start + 0.30s`.

### 8b. Chia khối để gửi LLM
- Cắt khối tại khoảng lặng `start[i+1] − end[i] > 2.0s`, **hoặc** khi khối đạt ~120 từ.
- Xử lý các khối **tuần tự** (đơn giản, đủ cho v1 — `translator.translate_segments` cũng chạy tuần tự theo batch). Song song hoá để sau nếu cần.

### 8c. LLM khôi phục dấu câu
- Client: tham số `llm_client`, mặc định `make_translation_client(cfg, ...)` (local_mac = Ollama qwen3).
- Prompt `prompts/restore_punctuation.yaml` (key `system`, `user`): yêu cầu **chỉ thêm dấu câu + viết hoa, KHÔNG thêm/bớt/đổi từ, không dịch**; trả JSON `{"text": "..."}` qua `response_format` json_schema strict.
- Track usage qua `tracker.add_llm_usage(prompt_tokens, completion_tokens)`.
- Retry transient + binary-split như `translator._translate_batch`.

### 8d. Re-align (token matching tuần tự)
- Tokenize `punctuated_text` thành token theo khoảng trắng (giữ dấu câu dính token).
- Duyệt song song với `words` gốc: chuẩn hoá token (strip dấu câu đầu/cuối + casefold) so với `word.text` chuẩn hoá. Vì LLM giữ nguyên chuỗi từ → khớp 1-1 theo thứ tự.
- Đóng câu khi token kết thúc bằng `.!?…。！？` (loại abbreviation theo whitelist sẵn có ở `transcriber._is_sentence_end`): tạo `Segment(start = start_của_từ_đầu_câu, end = end_của_từ_cuối_câu, text = câu_có_dấu_câu)`.
- Token dư cuối khối (không kết bằng dấu câu) → flush thành một segment.

### 8e. Fallback 🅑 (khi re-align lệch)
- Điều kiện: `abs(#token_LLM − #word_gốc) / #word_gốc > 0.15` (LLM lỡ sửa/thêm/bớt từ).
- Hành động: tạo segment theo **khối** (mỗi khối ở 8b thành 1 `Segment`, `start`/`end` từ word đầu/cuối khối, `text` = output LLM của khối). `split_into_sentences` (char-proportional) sẽ cắt câu. **Vẫn dùng caption, không rớt về Whisper.**

## 9. Interface & điểm tích hợp

| Nơi | Thay đổi |
|-----|----------|
| `pipeline/orchestrator.py` `DubOptions` | thêm `prefer_source_captions: bool = True` |
| `pipeline/orchestrator.py` `dub_video()` | thêm param `segments_override: list[Segment] \| None = None`; nhánh bỏ `extract_audio`+`transcribe`, emit progress `"Using source captions…"`, vẫn `merge_continuous_segments` + persist `"transcribed"` |
| `main.py` | `is_url(input)` + `opts.prefer_source_captions` → gọi `fetch_source_captions(url, source_language)`; truyền `segments_override`. Thêm flag `--no-source-captions` (mặc định bật); tải video vẫn diễn ra như cũ |
| `api/worker.py` | `_run_url_job` sau `_download_url` → `fetch_source_captions(url, ...)`; truyền `segments_override` xuống `_run_job`/`dub_video` |
| `api/models.py` + `api/routes/jobs.py` | field `prefer_source_captions: bool = True` ở request from-url |
| `api/static/index.html` | 1 checkbox "Ưu tiên phụ đề YouTube (nhanh hơn)", mặc định bật |
| `config/default.yaml` | `transcription.prefer_source_captions: true` |
| `prompts/restore_punctuation.yaml` | prompt mới (system/user) |

## 10. Xử lý lỗi (phân tầng — luôn có đường lui)

1. Input không phải URL (file local) → bỏ qua caption, Whisper.
2. `extract_info`/`_download_json3` lỗi, hoặc `_select_track` trả None → log cảnh báo, `fetch_source_captions` trả `None` → Whisper.
3. Parse ra 0 segment → `None` → Whisper.
4. LLM restore hỏng hoặc re-align lệch quá ngưỡng → fallback 🅑 (vẫn dùng caption).
5. `prefer_source_captions = False` (flag/config/checkbox) → luôn Whisper.

Mọi lỗi ở tầng caption đều **không** làm hỏng job — cùng lắm là quay về hành vi Whisper hiện tại.

## 11. Testing (không gọi YouTube thật)

Fixture json3 nhỏ + mock `yt_dlp.YoutubeDL.extract_info`, `urllib.request.urlopen`, và LLM client:
- `_select_track`: ưu tiên manual; chọn ASR gốc; **loại** track auto-translated.
- `_parse_manual`: cue → Segment, timestamp đúng.
- `_parse_auto_words`: gom word-level, bỏ `"\n"`, dedup rolling-overlap.
- `_restore_punctuation_and_align`: với LLM mock trả text có dấu câu, kiểm ranh giới câu + `start`/`end` khớp word-level.
- Fallback 🅑: khi token mismatch > 15%.
- `fetch_source_captions`: trả `None` khi không có track / 0 segment.
- Orchestrator: mở rộng `tests/test_orchestrator_artifacts.py` cho nhánh `segments_override` (mock translate/TTS) — xác nhận **không** gọi `transcribe`.

## 12. Rủi ro & đánh đổi

- **Chất lượng auto-caption** đôi khi kém Whisper large-v3 (sai thuật ngữ/tên riêng). Giảm thiểu: bước dịch đã có `asr_corrections` trong `default.yaml`; user có `--no-source-captions`.
- **LLM đổi từ** khi restore punctuation → fallback 🅑 đã xử lý.
- **Format json3 khác nhau giữa video** (rolling vs block). Dedup phòng thủ ở 8a; nếu vẫn lỗi parse → tầng lỗi #2/#3 đưa về Whisper.
- **Phụ thuộc mạng/biến động yt-dlp**: cô lập trong `captions.py`, mọi lỗi → fallback Whisper.

## 13. Giá trị tham số (chốt, tránh mơ hồ)

- Dedup overlap ngưỡng thời gian: `0.10s`.
- `end` từ cuối: `+0.30s`.
- Chia khối LLM: khoảng lặng `> 2.0s` hoặc `~120 từ`.
- Ngưỡng fallback re-align: lệch token `> 15%`.
- Format caption: `json3`.
- Mặc định: `prefer_source_captions = true`.
