import numpy as np
import pytest

from pipeline import diarizer
from pipeline.diarizer import Turn
from pipeline.transcriber import Segment


def _embs(centers, n_each=5, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for c in centers:
        for _ in range(n_each):
            v = c + rng.normal(0, noise, size=c.shape)
            out.append(v / np.linalg.norm(v))
    return np.array(out)


def test_cluster_two_speakers_by_threshold():
    pytest.importorskip("scipy")
    a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    labels = diarizer.cluster_embeddings(_embs([a, b]), threshold=0.5)
    assert len(set(labels[:5])) == 1 and len(set(labels[5:])) == 1 and labels[0] != labels[5]


def test_cluster_fixed_num_speakers_and_cap():
    pytest.importorskip("scipy")
    a, b, c = np.eye(3)
    labels = diarizer.cluster_embeddings(_embs([a, b, c]), num_speakers=2)
    assert len(set(labels)) == 2
    labels = diarizer.cluster_embeddings(_embs([a, b, c]), threshold=0.1, max_speakers=2)
    assert len(set(labels)) <= 2


def test_relabel_by_first_appearance():
    assert diarizer.relabel_by_first_appearance([7, 7, 2, 7, 2, 9]) == [
        "SPEAKER_00",
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_02",
    ]


def test_assign_by_overlap_and_nearest():
    turns = [Turn(0.0, 5.0, "A"), Turn(5.0, 10.0, "B")]
    segs = [
        Segment(id=0, start=1.0, end=4.0, text="x"),
        Segment(id=1, start=4.5, end=7.0, text="y"),
        Segment(id=2, start=12.0, end=13.0, text="z"),
    ]
    assert diarizer.assign_by_overlap(segs, turns) == ["A", "B", "B"]


def test_label_segments_ecapa_with_fake_encoder(tmp_path, monkeypatch):
    pytest.importorskip("scipy")  # label_segments clusters the (fake) embeddings for real
    import soundfile as sf

    sr = 16000
    sf.write(tmp_path / "a.wav", np.zeros(sr * 12, dtype=np.float32), sr)
    segs = [Segment(id=i, start=i * 2.0, end=i * 2.0 + 1.5, text="s") for i in range(6)]

    def fake_loader(model, device):
        # embed(wav_crop, sr, t0) -> embedding by time: first 3 segs -> e1, rest -> e2
        def embed(wav_crop, sr, t0):
            return np.array([1.0, 0.0]) if t0 < 6.0 else np.array([0.0, 1.0])

        return embed

    monkeypatch.setattr(diarizer, "_load_ecapa_embedder", fake_loader)
    labels = diarizer.label_segments(str(tmp_path / "a.wav"), segs, backend="ecapa", threshold=0.5)
    assert labels == ["SPEAKER_00"] * 3 + ["SPEAKER_01"] * 3
