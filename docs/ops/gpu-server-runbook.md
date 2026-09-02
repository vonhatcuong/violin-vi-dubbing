# Runbook: dựng server GPU chạy lồng tiếng EN → VI (fully local)

Tài liệu này ghi lại **toàn bộ** cách máy chủ GPU hiện tại được dựng và vận hành,
đủ để dựng lại trên một server mới (Vast.ai, RunPod, máy tự host…) mà không cần
nhớ gì thêm. Mọi lệnh đã được chạy thật trên server tham chiếu ngày 2026-09-02.

## 0. Server tham chiếu (đang dùng)

| Thành phần | Giá trị |
|---|---|
| Nhà cung cấp | Vast.ai, SSH `ssh -p 41620 root@<ip>` (cổng đổi theo instance) |
| OS / kernel | Ubuntu 24.04.4 LTS / 6.8.0 |
| GPU | 2× NVIDIA GeForce RTX 3090 24 GB, driver 580.159.03, CUDA driver 13.0, toolkit 12.8 |
| Python | 3.13 (venv do `uv` tạo) |
| uv / ffmpeg / ollama | uv 0.12.7 / ffmpeg 6.1.1 / ollama 0.33.2 |
| Ổ đĩa persistent | `/workspace` (100 GB, dùng ~53 GB) |
| Repo | `/workspace/violin` (nhánh `feat/local-vi-dubbing`, origin = `github.com/vonhatcuong/violin-vi-dubbing`, private) |

Phiên bản Python package đang chạy (đã kiểm chứng tương thích với nhau):

```
torch==2.8.0+cu128   torchaudio==2.8.0+cu128   faster-whisper==1.2.1   ctranslate2==4.7.1
vieneu==3.3.0        transformers==4.57.6      speechbrain==1.1.1      scipy==1.18.1
soundfile==0.13.1    numpy==2.4.4              openai==2.32.0          yt-dlp==2026.8.19
```

Bố trí thư mục trên `/workspace`:

```
/workspace/violin            repo + .venv (7.6 GB)
/workspace/ollama            OLLAMA_MODELS — gemma4:31b (19 GB)
/workspace/.hf_home          HF_HOME — cache model HF (3.4 GB):
                               Systran/faster-whisper-large-v3 (2.9 GB)
                               pnnbao-ump/VieNeu-TTS-v3-Turbo (305 MB)
                               OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano (85 MB, codec của VieNeu)
                               speechbrain/spkrec-ecapa-voxceleb (85 MB, tách người nói)
/workspace/samples           video nguồn (.mp4) tải bằng yt-dlp
/workspace/outN              kết quả từng lần chạy (mp4, srt, *.segments.json, fit.units.json, log)
/workspace/config_run*.yaml  preset override của từng lần chạy (xem §4)
/workspace/ollama.log        log Ollama
```

## 1. Yêu cầu phần cứng cho server mới

- **VRAM**: gemma4:31b (Q4) chiếm ~21 GB khi chạy (model 19 GB + KV cache 8k).
  ASR (faster-whisper large-v3 fp16) + VieNeu batch cần thêm ~5–6 GB.
  → Cần **2 GPU ≥ 24 GB** (Ollama một card, ASR/TTS card kia) **hoặc 1 GPU ≥ 40 GB**.
  Nếu chỉ có 1× 24 GB: dùng `gemma4:12b` (≈8 GB) cho bước dịch; chất lượng thấp hơn 31b.
- **Đĩa**: ≥ 60 GB trống (venv 8 GB + Ollama 19 GB + HF cache 3.5 GB + video/kết quả).
- **RAM**: ≥ 32 GB. **CPU**: ≥ 8 lõi (ffmpeg ghép video 1080p dùng 16 worker).
- Mạng ra ngoài chỉ cần **lúc cài** (tải model); lúc chạy hoàn toàn offline.

## 2. Cài đặt từ đầu (chạy tuần tự bằng root)

### 2.1 Gói hệ thống

```bash
apt-get update && apt-get install -y ffmpeg git curl ca-certificates
nvidia-smi -L        # phải thấy GPU; ghi lại UUID từng card (xem §3)
```

### 2.2 uv + repo + venv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
mkdir -p /workspace && cd /workspace
# repo private: cần SSH key của tài khoản GitHub (ssh-keygen, thêm vào github.com/settings/keys)
git clone git@github.com:vonhatcuong/violin-vi-dubbing.git violin
cd violin && git checkout feat/local-vi-dubbing

uv venv --python 3.13
uv sync --extra local-gpu              # faster-whisper, vieneu, transformers 4.57.6, speechbrain, scipy…
# torch CUDA 12.8 mà engine GPU của VieNeu cần (uv sync cài bản CPU mặc định → thay bằng bản cu128)
uv pip install "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install -U yt-dlp               # tải video YouTube
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.8.0+cu128 True
```

Không cần file `.env` cho stack local (provider `ollama` trỏ localhost không đòi API key).

### 2.3 Biến môi trường (đặt vào `~/.bashrc` để cache nằm trên đĩa persistent)

```bash
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/workspace/.hf_home          # cache Whisper / VieNeu / ECAPA
export OLLAMA_MODELS=/workspace/ollama      # cache model Ollama
export PATH="$HOME/.local/bin:$PATH"
EOF
source ~/.bashrc
```

### 2.4 Ollama + Gemma 4

```bash
curl -fsSL https://ollama.com/install.sh | sh          # cài vào /usr/local/bin/ollama
# Khởi động Ollama TRÊN MỘT CARD CỐ ĐỊNH (UUID, không dùng chỉ số 0/1 — xem §3)
CUDA_VISIBLE_DEVICES=GPU-<uuid-A> OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS=/workspace/ollama \
OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KEEP_ALIVE=30m OLLAMA_NUM_PARALLEL=2 \
setsid nohup ollama serve > /workspace/ollama.log 2>&1 < /dev/null &

ollama pull gemma4:31b                                  # ~19 GB, một lần
curl -s localhost:11434/api/tags | head -c 300          # phải liệt kê gemma4:31b
```

Vì sao từng biến quan trọng:

- `OLLAMA_CONTEXT_LENGTH=8192`: mặc định Gemma 4 mở context 262k → tràn nửa model sang CPU, còn 3 tok/s.
- `OLLAMA_KEEP_ALIVE=30m`: giữ model trên GPU giữa các bước; lần gọi đầu sau khi idle mất 20–40 s nạp lại.
- `OLLAMA_NUM_PARALLEL=2` khớp `translation.parallel_batches: 2` trong preset.
- `OLLAMA_KV_CACHE_TYPE=q8_0` + flash attention: giảm VRAM KV cache.

Kiểm tra model nằm 100 % trên GPU: `ollama ps` → cột PROCESSOR phải là `100% GPU`.

### 2.5 Tải sẵn model ASR / TTS / diarization (tự tải lần chạy đầu, nhưng tải trước sẽ nhanh hơn)

```bash
cd /workspace/violin
CUDA_VISIBLE_DEVICES=GPU-<uuid-B> .venv/bin/python - <<'EOF'
from faster_whisper import WhisperModel
WhisperModel("large-v3", device="cuda", compute_type="float16")              # 2.9 GB
from vieneu import Vieneu
Vieneu(backend="pytorch", device="cuda")                                       # VieNeu v3 Turbo + MOSS codec
from speechbrain.inference.speaker import EncoderClassifier
EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda"})
print("models cached under", __import__("os").environ["HF_HOME"])
EOF
```

## 3. Gán GPU bằng UUID (bắt buộc khi có 2 card giống nhau)

`CUDA_VISIBLE_DEVICES=0` / `=1` **không ổn định**: mỗi tiến trình có thể đánh số 2 card RTX 3090
theo thứ tự khác nhau, từng gây OOM vì Ollama và Whisper/VieNeu rơi vào cùng một card.

```bash
nvidia-smi -L
# GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-a04c97d8-…)   → card A: Ollama
# GPU 1: NVIDIA GeForce RTX 3090 (UUID: GPU-8ff58980-…)   → card B: ASR + TTS + diarization
```

Quy ước trên server hiện tại: **card A = Ollama, card B = pipeline**. Mọi lệnh `main.py` đều
đặt `CUDA_VISIBLE_DEVICES=GPU-<uuid-B>`. Kiểm tra lúc chạy bằng
`nvidia-smi --query-gpu=uuid,memory.used --format=csv` (card A ~21 GB, card B ~5 GB).

## 4. Preset cấu hình

Preset gốc trong repo: `config/local_gpu.yaml` (đã có: Whisper large-v3 fp16 CUDA, Ollama gemma4:31b,
VieNeu batch CUDA, `fit` bật với pipelined + `min_fill 0.6`, tách câu dài 12 s, phụ đề tiếng Anh sidecar,
giọng mặc định Phạm Tuyên / Ngọc Huyền, `batch_size 16`, `parallel_batches 2`).

Bản override đang dùng cho các lần chạy gần nhất (`/workspace/config_run10.yaml`) chỉ khác preset ở
`transcription.parallel_workers: 2` (2 chunk Whisper song song) và `merge.max_duration: 15`.
Tạo lại trên server mới:

```bash
cd /workspace/violin && .venv/bin/python - <<'EOF'
import yaml
cfg = yaml.safe_load(open("config/local_gpu.yaml"))
cfg["transcription"]["parallel_workers"] = 2
cfg.setdefault("merge", {})["max_duration"] = 15
yaml.safe_dump(cfg, open("/workspace/config_gpu.yaml", "w"), allow_unicode=True, sort_keys=False)
EOF
```

Kích thước file ra: preset mặc định `merge_video.preset: ultrafast, crf: 23` cho bitrate cao (~1,6 Mbps ở 640×480,
~3,6 Mbps ở 1080p). Để file nhẹ hơn 3–4 lần mà tốc độ gần như không đổi, đặt trong override:
`merge_video.crf: 26` và `merge_video.preset: veryfast`.

Các khóa hay chỉnh: `voices.default_male/female`, `voices.speaker_voices`, `fit.sec_per_syllable`
(đo lại bằng `scripts/calibrate_voice.py --voice "<tên>" --config /workspace/config_gpu.yaml`;
Phạm Tuyên 0.227, Ngọc Huyền 0.230, Thanh Bình 0.230 s/âm tiết), `translation.batch_size`
(16: ít lỗi sai số câu hơn 32), `diarization.threshold` / `max_speakers`.

## 5. Chạy lồng tiếng

### 5.1 Tải video nguồn

```bash
cd /workspace/samples
/workspace/violin/.venv/bin/yt-dlp \
  -f "bv*[ext=mp4][vcodec^=avc1][height<=1080]+ba[ext=m4a]/bv*[height<=1080]+ba/b" \
  --merge-output-format mp4 --extractor-args "youtube:player_client=web_safari,android,default" \
  -o "/workspace/samples/<id>.mp4" "https://www.youtube.com/watch?v=<id>"
```

Gặp `HTTP Error 403` → `uv pip install -U yt-dlp` rồi chạy lại đúng lệnh trên (AV1 format 399 hay bị chặn; ép avc1).

### 5.2 Lệnh chạy (tách tiến trình khỏi phiên SSH)

```bash
cd /workspace/violin && mkdir -p /workspace/out_<tên>
# 1 người nói (bài giảng)
CUDA_VISIBLE_DEVICES=GPU-<uuid-B> setsid nohup .venv/bin/python main.py \
  /workspace/samples/<id>.mp4 /workspace/out_<tên>/<tên>_vi.mp4 \
  --language Vietnamese --config /workspace/config_gpu.yaml \
  > /workspace/out_<tên>/run.log 2>&1 < /dev/null &

# nhiều người nói (podcast, talk có host): thêm --speakers auto (hoặc --speakers 2)
#   ... --speakers auto --voice-map "SPEAKER_00=Phạm Tuyên,SPEAKER_01=Ngọc Huyền"   # tùy chọn ép giọng
# phụ đề burn cứng (mặc định KHÔNG burn, chỉ xuất .srt): thêm --burn-subtitles
# giọng khác cho toàn video: --voice "Minh Đức"
```

Theo dõi: `tail -f /workspace/out_<tên>/run.log` — các mốc `[1/5] … [5/5]`, `[pipeline] N/M units done`,
`[speakers] N speakers → {...}`. Tiến trình còn chạy: `pgrep -f "^\.venv/bin/python main\.py"`.

### 5.3 Kết quả trong thư mục out

```
<tên>_vi.mp4                        video đã lồng tiếng (nhạc nền/giọng gốc còn 2 %)
<tên>_vi.srt                        phụ đề tiếng Anh (sidecar, đã map theo thời gian video mới)
<tên>_vi.transcript.txt             bản dịch dạng text
<tên>_vi.transcribed.segments.json  ASR (kèm word timestamps)
<tên>_vi.sentences.segments.json    câu gốc dùng cho phụ đề (có speaker)
<tên>_vi.diarized.segments.json     (khi --speakers) nhãn SPEAKER_xx từng câu
<tên>_vi.voices.json                (khi --speakers) giọng gán cho từng speaker
<tên>_vi.translated.segments.json   checkpoint bản dịch (ghi dần từng batch)
<tên>_vi.fitted.segments.json       câu sau khi khớp thời lượng
<tên>_vi.fit.units.json             chi tiết từng unit: slot, tts_dur, strategy (natural/over/slowed/shortened)
```

Xem phụ đề mềm không cần re-encode:
`ffmpeg -i <tên>_vi.mp4 -i <tên>_vi.srt -c copy -c:s mov_text -metadata:s:s:0 language=eng <tên>_vi_sub.mp4`.

Chạy lại từ checkpoint (sửa tay bản dịch rồi tổng hợp lại):
`.venv/bin/python resume_from_segments.py --input /workspace/samples/<id>.mp4 --segments /workspace/out_<tên>/<tên>_vi.translated.segments.json --output /workspace/out_<tên>/<tên>_vi2.mp4 --language Vietnamese --config /workspace/config_gpu.yaml`
(chỉ nhận stage `transcribed | translated | fitted`; đọc `voices.json` bên cạnh nếu có).

### 5.4 Thời gian tham chiếu (2× RTX 3090, đo một lần, không phải benchmark)

| Video | Lệnh | ASR | Dịch + TTS | Tổng pipeline | Ghép video (ngoài tracker) |
|---|---|---|---|---|---|
| Bài giảng 21 phút (1267 s) | mặc định | 1m37 | 2m41 | 4m23 | ~2 phút |
| Phỏng vấn 2 người 134 s | `--speakers auto` | ~20 s | ~25 s | 52 s | ~15 s |
| Talk 74 phút (4456 s) | `--speakers auto` | ~6 phút | ~10 phút | ~17 phút | ~7 phút |

Lần gọi Ollama đầu tiên sau khi idle > 30 phút tốn thêm 20–40 s nạp model.

## 6. Copy kết quả về máy cá nhân

Mạng Vast.ai → máy cá nhân chỉ ~1 MB/s, nên nén trước trên server (NVENC, giữ độ phân giải nhưng tối đa 720p),
rồi chỉ kéo bản nén:

```bash
# trên server: ~1–2 phút cho 40 phút video; ffprobe để chắc file hoàn chỉnh trước khi copy
ffmpeg -hide_banner -loglevel error -y -hwaccel cuda -i <tên>_vi.mp4 -vf "scale=-2:'min(720,ih)'" \
  -c:v h264_nvenc -preset p4 -cq 30 -c:a aac -b:a 96k <tên>_vi_720p.mp4 && ffprobe -v error -show_entries format=duration -of csv=p=0 <tên>_vi_720p.mp4
```

```bash
# từ máy Mac
D=~/…/violin/output/e2e_server/<tên>; mkdir -p "$D"
rsync -a --partial -e "ssh -p <port>" root@<ip>:/workspace/out_<tên>/<tên>_vi_720p.mp4 \
  :/workspace/out_<tên>/<tên>_vi.srt :/workspace/out_<tên>/<tên>_vi.fit.units.json "$D/"
```

Chạy hàng loạt (playlist): script mẫu chạy tuần tự, bỏ qua bài đã có `DONE`, tải từng video bằng yt-dlp và ghi
`batch.log` — bản sao trong repo: `scripts/ops/batch_playlist_example.sh` (server) và `scripts/ops/sync_loop_example.sh` (máy cá nhân) (khung: vòng `for id in $IDS`, `yt-dlp` →
`main.py` → `touch DONE`). Kết hợp với một vòng lặp rsync trên máy cá nhân chỉ kéo thư mục có `DONE`.

Giải phóng đĩa server: mỗi bài giảng để lại ~0,5–1 GB (bản gốc + 720p + audio gốc + video nguồn); ổ 100 GB đầy sau ~60 bài.
Cách vận hành đã ổn định: trên server chạy `scripts/ops/encode720_daemon_example.sh` (nén 720p ngay khi bài xong, ghi
`<file>.ok`); trên máy cá nhân chạy `scripts/ops/sync_loop_example.sh` (MỘT kết nối SSH mỗi chu kỳ để lấy danh sách `.ok`,
rồi mỗi bài một rsync — mở nhiều kết nối SSH liên tiếp sẽ bị Vast.ai reset). Vòng kéo KHÔNG xoá gì trên server; chỉ xoá tay
sau khi đã sao lưu kết quả ở nơi thứ hai (bài học 2026-09-02: một lỗi xoá thư mục ở máy cá nhân làm mất 25 bài đã dọn trên server):
`rm -f /workspace/out_<khoá>/<tag>/{*_vi.mp4,*_720p.mp4,*_original.m4a} /workspace/samples/<khoá>/<id>.mp4`.

## 7. Chuyển sang server mới: sao lưu gì, tải lại gì

| Mục | Kích thước | Sao lưu hay tải lại? |
|---|---|---|
| `/workspace/violin` (code) | nhỏ | tải lại từ GitHub (`git clone`), `.venv` dựng lại theo §2.2 (~10 phút) |
| `/workspace/ollama` | 19 GB | `ollama pull gemma4:31b` tải lại (~5–15 phút tùy mạng) hoặc rsync sang |
| `/workspace/.hf_home` | 3.4 GB | tự tải lại (§2.5) hoặc rsync sang |
| `/workspace/config_run*.yaml` | KB | **sao lưu** (hoặc tạo lại theo §4) |
| `/workspace/samples`, `/workspace/out*` | GB | chỉ giữ cái cần |

Rsync trực tiếp giữa 2 server: `rsync -a -e "ssh -p <port_cũ>" root@<ip_cũ>:/workspace/{ollama,.hf_home} /workspace/`.

Sau khi dựng xong, chạy thử trên clip ngắn trước (ví dụ clip phỏng vấn 134 s với `--speakers auto`):
kỳ vọng ~1 phút, `voices.json` có 2 giọng khác nhau, `fit.units.json` không có unit dài quá 12 s.

## 8. Sự cố đã gặp và cách xử lý

| Triệu chứng | Nguyên nhân | Xử lý |
|---|---|---|
| `CUDA out of memory` khi TTS/Whisper | Ollama và pipeline rơi cùng card do dùng chỉ số GPU | Dùng UUID (§3) |
| Dịch cực chậm (3 tok/s), `ollama ps` báo `xx% CPU` | context 262k mặc định tràn RAM | `OLLAMA_CONTEXT_LENGTH=8192`, khởi động lại Ollama |
| Mỗi câu dịch ~13 s | Gemma 4 bật "thinking" | preset đã gửi `reasoning_effort: none` (`translation.reasoning_effort`) |
| `⚠ Count mismatch … retrying` rồi `Splitting failed batch` | LLM trả sai số câu trong batch lớn | `translation.batch_size: 16` (đã là mặc định GPU) |
| Unit dài 30–160 s trong `fit.units.json` | Whisper gộp câu qua khoảng lặng dài | `transcription.max_sentence_seconds: 12`, `long_gap_seconds: 2.0` (mặc định) |
| Giọng nuốt phụ âm đầu câu | preset VieNeu "Thanh Bình" xuất audio bắt đầu ngay mẫu 0 | dùng Phạm Tuyên (mặc định), tránh Thanh Bình |
| `--speakers auto` ra 3–4 speaker trong clip 2 người | câu 0,5 s cho embedding yếu | `diarization.min_cluster_segments/seconds` (mặc định 3 / 3.0) tự gộp |
| `ModuleNotFoundError: torch/speechbrain` khi `--speakers` | thiếu extras GPU | `uv sync --extra local-gpu` + torch cu128 (§2.2) |
| yt-dlp `HTTP Error 403` | format AV1 / client cũ | nâng cấp yt-dlp, ép avc1 + `player_client` (§5.1) |
| Ollama không nhận key | không cần | stack local không dùng `.env`; đừng đặt `OLLAMA_API_KEY` trỏ cloud |

## 9. Ghi chú vận hành trên Vast.ai

- Instance có Jupyter chạy sẵn ở cổng 8080 (`ssh -L 8080:localhost:8080` nếu cần).
- Chỉ `/workspace` là persistent giữa các lần start/stop; `~/.cache`, `/tmp` có thể mất → vì vậy `HF_HOME` và `OLLAMA_MODELS` đặt trong `/workspace`.
- Sau khi instance khởi động lại phải **khởi động lại Ollama** bằng lệnh ở §2.4 (không có systemd).
- Đổi instance = đổi cổng SSH; cập nhật `-p <port>` trong mọi lệnh ssh/rsync.
