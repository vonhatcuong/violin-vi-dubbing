from pipeline import config as pipeline_config
from pipeline import transcriber
from pipeline.transcriber import Segment


def test_chunk_offset_shifts_words_too(monkeypatch):
    pipeline_config.load()

    def fake_split_audio(audio_path, chunk_seconds=600):
        return [("a.wav", 0.0), ("b.wav", 600.0)]

    def fake_transcribe_single(audio_path, client, model):
        return [Segment(id=0, start=1.0, end=1.9, text="Hello there.",
                         words=[["Hello", 1.0, 1.4], ["there.", 1.5, 1.9]])]

    monkeypatch.setattr(transcriber, "split_audio", fake_split_audio)
    monkeypatch.setattr(transcriber, "_transcribe_single", fake_transcribe_single)

    out = transcriber.transcribe("x.wav", client=object())

    assert out[0].start == 1.0
    assert out[0].words == [["Hello", 1.0, 1.4], ["there.", 1.5, 1.9]]

    assert out[1].start == 601.0
    assert out[1].words == [["Hello", 601.0, 601.4], ["there.", 601.5, 601.9]]
