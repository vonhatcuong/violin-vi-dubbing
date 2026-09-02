# Voice bank (vi)

Mỗi giọng = 1 clip WAV 24 kHz mono dài 5–10 s, nói rõ, không nhạc nền + transcript chính xác (viết thường).
Tạo bằng:

    uv run scripts/make_ref_clip.py --source path/to/audio_or_video --start 12.0 --end 20.0 --name nam-1 --gender male
    # thêm --ref-text "..." nếu không muốn tự transcribe bằng faster-whisper

`catalog.yaml` được cập nhật tự động. Không commit clip có bản quyền của người khác.
