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


def test_absorb_small_clusters_reassigns_to_nearest_large():
    rng = np.random.default_rng(1)
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])

    def near(center, n, noise=0.02):
        out = []
        for _ in range(n):
            v = center + rng.normal(0, noise, size=center.shape)
            out.append(v / np.linalg.norm(v))
        return out

    cluster_a = near(a, 5)
    cluster_b = near(b, 4)
    singleton_near_a = near(a, 1)
    small_near_b = near(b, 2)

    embs = np.array(cluster_a + cluster_b + singleton_near_a + small_near_b)
    labels = [0] * 5 + [1] * 4 + [2] * 1 + [3] * 2
    durations = [1.0] * 5 + [1.0] * 4 + [1.0] * 1 + [0.45, 0.45]  # small_near_b totals 0.9 s

    out = diarizer.absorb_small_clusters(labels, embs, durations, min_segments=3, min_seconds=3.0)

    assert out[:5] == [0] * 5
    assert out[5:9] == [1] * 4
    assert out[9] == 0  # singleton near A absorbed into A
    assert out[10:12] == [1, 1]  # 2-member cluster near B absorbed into B


def test_absorb_small_clusters_all_small_returns_unchanged():
    embs = np.array([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]])
    labels = [0, 0, 1, 1]
    durations = [10.0, 10.0, 10.0, 10.0]  # long enough, but each cluster has only 2 members
    out = diarizer.absorb_small_clusters(labels, embs, durations, min_segments=3, min_seconds=3.0)
    assert out == labels


def test_absorb_small_clusters_min_seconds_alone_triggers():
    rng = np.random.default_rng(2)
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    cluster_a = [v / np.linalg.norm(v) for v in (a + rng.normal(0, 0.01, 3) for _ in range(5))]
    cluster_b = [v / np.linalg.norm(v) for v in (b + rng.normal(0, 0.01, 3) for _ in range(3))]

    embs = np.array(cluster_a + cluster_b)
    labels = [0] * 5 + [1] * 3
    # cluster 1 has 3 members (not < min_segments) but only 1.2 s total (< min_seconds)
    durations = [2.0] * 5 + [0.4] * 3

    out = diarizer.absorb_small_clusters(labels, embs, durations, min_segments=3, min_seconds=3.0)
    assert out == [0] * 5 + [0, 0, 0]


def test_absorb_small_clusters_min_segments_alone_triggers():
    rng = np.random.default_rng(3)
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    cluster_a = [v / np.linalg.norm(v) for v in (a + rng.normal(0, 0.01, 3) for _ in range(5))]
    cluster_b = [v / np.linalg.norm(v) for v in (b + rng.normal(0, 0.01, 3) for _ in range(2))]

    embs = np.array(cluster_a + cluster_b)
    labels = [0] * 5 + [1] * 2
    # cluster 1 has 20 s total (not < min_seconds) but only 2 members (< min_segments)
    durations = [2.0] * 5 + [10.0] * 2

    out = diarizer.absorb_small_clusters(labels, embs, durations, min_segments=3, min_seconds=3.0)
    assert out == [0] * 5 + [0, 0]


def test_absorb_small_clusters_single_cluster_unchanged():
    embs = np.array([[1.0, 0.0], [0.9, 0.1], [1.0, 0.0]])
    labels = [0, 0, 0]
    durations = [0.1, 0.1, 0.1]
    out = diarizer.absorb_small_clusters(labels, embs, durations, min_segments=3, min_seconds=3.0)
    assert out == labels


def test_relabel_by_first_appearance():
    assert diarizer.relabel_by_first_appearance([7, 7, 2, 7, 2, 9]) == [
        "SPEAKER_00",
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_02",
    ]


def test_crop_bounds_interior_short_segment_centred():
    sr = 16000
    n_samples = 10 * sr  # 10 s file
    s0, s1 = diarizer._crop_bounds(5.0, 5.2, sr, n_samples)
    assert s1 - s0 == int(round(0.5 * sr))
    # centred on the original [5.0, 5.2] window's midpoint (5.1 s)
    mid = (s0 + s1) / 2
    assert mid == pytest.approx(5.1 * sr)


def test_crop_bounds_overshoots_file_end():
    sr = 16000
    n_samples = int(round(2.0 * sr))  # 2.0 s file
    s0, s1 = diarizer._crop_bounds(1.9, 5.0, sr, n_samples)
    assert (s0, s1) == (int(round(1.5 * sr)), int(round(2.0 * sr)))


def test_crop_bounds_at_file_start():
    sr = 16000
    n_samples = 10 * sr  # long enough file
    s0, s1 = diarizer._crop_bounds(0.0, 0.1, sr, n_samples)
    assert (s0, s1) == (0, int(round(0.5 * sr)))


def test_crop_bounds_file_shorter_than_min():
    sr = 16000
    n_samples = int(round(0.25 * sr))  # file shorter than 0.5 s
    s0, s1 = diarizer._crop_bounds(0.1, 0.15, sr, n_samples)
    assert (s0, s1) == (0, n_samples)


def test_crop_bounds_segment_already_long_enough():
    sr = 16000
    n_samples = 10 * sr
    s0, s1 = diarizer._crop_bounds(1.0, 2.0, sr, n_samples)
    assert (s0, s1) == (int(round(1.0 * sr)), int(round(2.0 * sr)))


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


def test_label_segments_ecapa_absorbs_small_cluster(tmp_path, monkeypatch):
    import soundfile as sf

    sr = 16000
    sf.write(tmp_path / "a.wav", np.zeros(sr * 14, dtype=np.float32), sr)
    segs = [Segment(id=i, start=i * 2.0, end=i * 2.0 + 1.5, text="s") for i in range(6)]
    segs.append(Segment(id=6, start=12.5, end=13.0, text="s"))  # 0.5 s spurious segment

    def fake_loader(model, device):
        # first 3 segs near [1,0] (speaker A), next 3 near [0,1] (speaker B), last near A
        def embed(wav_crop, sr, t0):
            if t0 < 6.0:
                return np.array([1.0, 0.0])
            elif t0 < 12.0:
                return np.array([0.0, 1.0])
            return np.array([0.9, 0.1])

        return embed

    def fake_cluster(embs, num_speakers=None, max_speakers=4, threshold=0.65):
        # 3 clusters: 0 and 1 are real speakers, 2 is a 0.5 s singleton artefact
        return [0, 0, 0, 1, 1, 1, 2]

    monkeypatch.setattr(diarizer, "_load_ecapa_embedder", fake_loader)
    monkeypatch.setattr(diarizer, "cluster_embeddings", fake_cluster)

    labels = diarizer.label_segments(str(tmp_path / "a.wav"), segs, backend="ecapa")
    assert len(set(labels)) == 2


def test_label_segments_ecapa_fixed_num_speakers_skips_absorption(tmp_path, monkeypatch):
    pytest.importorskip("scipy")
    import soundfile as sf

    sr = 16000
    sf.write(tmp_path / "a.wav", np.zeros(sr * 7, dtype=np.float32), sr)
    # 5 "majority" segments near [1,0], 2 "minority" segments near [0,1] — the minority
    # cluster (2 < default min_cluster_segments=3) would normally be absorbed into the
    # majority in auto mode, but num_speakers=2 is a user-fixed count.
    segs = [Segment(id=i, start=i * 1.0, end=i * 1.0 + 0.8, text="s") for i in range(7)]

    def fake_loader(model, device):
        def embed(wav_crop, sr, t0):
            return np.array([1.0, 0.0]) if t0 < 5.0 else np.array([0.0, 1.0])
        return embed

    monkeypatch.setattr(diarizer, "_load_ecapa_embedder", fake_loader)

    labels = diarizer.label_segments(str(tmp_path / "a.wav"), segs, backend="ecapa", num_speakers=2)
    assert labels == ["SPEAKER_00"] * 5 + ["SPEAKER_01"] * 2


def test_label_segments_ecapa_auto_mode_never_absorbs_down_to_one_speaker(tmp_path, monkeypatch):
    import soundfile as sf

    sr = 16000
    sf.write(tmp_path / "a.wav", np.zeros(sr * 11, dtype=np.float32), sr)
    segs = [Segment(id=i, start=i * 2.0, end=i * 2.0 + 1.5, text="s") for i in range(5)]
    segs.append(Segment(id=5, start=10.5, end=11.0, text="s"))  # 0.5 s minority speaker

    def fake_loader(model, device):
        def embed(wav_crop, sr, t0):
            return np.array([1.0, 0.0]) if t0 < 10.0 else np.array([0.0, 1.0])
        return embed

    def fake_cluster(embs, num_speakers=None, max_speakers=4, threshold=0.65):
        # 1 large cluster (5 members) + 1 tiny cluster (1 member) — absorption would
        # normally fold the tiny cluster into the large one, leaving a single speaker.
        return [0, 0, 0, 0, 0, 1]

    monkeypatch.setattr(diarizer, "_load_ecapa_embedder", fake_loader)
    monkeypatch.setattr(diarizer, "cluster_embeddings", fake_cluster)

    labels = diarizer.label_segments(str(tmp_path / "a.wav"), segs, backend="ecapa")
    assert len(set(labels)) == 2
    assert labels == ["SPEAKER_00"] * 5 + ["SPEAKER_01"]


def test_label_segments_ecapa_all_embeddings_fail_raises(tmp_path, monkeypatch):
    import soundfile as sf

    sr = 16000
    sf.write(tmp_path / "a.wav", np.zeros(sr * 3, dtype=np.float32), sr)
    segs = [Segment(id=i, start=i * 1.0, end=i * 1.0 + 0.8, text="s") for i in range(3)]

    def fake_loader(model, device):
        def embed(wav_crop, sr, t0):
            raise RuntimeError("boom")
        return embed

    monkeypatch.setattr(diarizer, "_load_ecapa_embedder", fake_loader)

    with pytest.raises(RuntimeError, match="3 speaker embeddings failed"):
        diarizer.label_segments(str(tmp_path / "a.wav"), segs, backend="ecapa")


def test_label_segments_ecapa_partial_embedding_failure_warns_and_inherits(tmp_path, monkeypatch, capsys):
    pytest.importorskip("scipy")
    import soundfile as sf

    sr = 16000
    sf.write(tmp_path / "a.wav", np.zeros(sr * 3, dtype=np.float32), sr)
    segs = [Segment(id=i, start=i * 1.0, end=i * 1.0 + 0.8, text="s") for i in range(3)]

    def fake_loader(model, device):
        def embed(wav_crop, sr, t0):
            if 0.9 < t0 < 1.1:
                raise RuntimeError("fail seg 1")
            return np.array([1.0, 0.0])
        return embed

    monkeypatch.setattr(diarizer, "_load_ecapa_embedder", fake_loader)

    labels = diarizer.label_segments(str(tmp_path / "a.wav"), segs, backend="ecapa")
    assert labels[1] == labels[0]  # failed segment inherits the previous label

    out = capsys.readouterr().out
    assert "1 speaker embedding(s) failed" in out
    assert "fail seg 1" in out
