"""Sentence-level speaker labelling.

Default backend `ecapa`: crop each ASR sentence from the (16 kHz mono) audio
file, embed it with speechbrain's ECAPA-TDNN speaker encoder, cluster the
embeddings (agglomerative, cosine, average linkage) and relabel clusters
`SPEAKER_00..` by order of first appearance.

Optional backend `pyannote`: run a full pyannote diarization pipeline and
assign each sentence to the speaker turn it overlaps most with.

Heavy imports (torch, speechbrain, pyannote) are lazy — this module is
importable, and its pure helpers are testable, without those packages
installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from .transcriber import Segment


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


def cluster_embeddings(
    embs: np.ndarray,
    num_speakers: int | None = None,
    max_speakers: int = 4,
    threshold: float = 0.65,
) -> list[int]:
    """Agglomerative clustering (average linkage, cosine distance) over L2-normalized embeddings.

    `num_speakers` given → cut the tree to exactly that many clusters
    (`fcluster(..., criterion="maxclust")`). Otherwise cut by cosine-distance
    `threshold` (`criterion="distance"`), then collapse further to at most
    `max_speakers` clusters if that produced more.
    """
    n = len(embs)
    if n == 0:
        return []
    if n == 1 or num_speakers == 1:
        return [0] * n

    from scipy.cluster.hierarchy import fcluster, linkage

    z = linkage(embs, method="average", metric="cosine")
    if num_speakers is not None:
        labels = fcluster(z, t=num_speakers, criterion="maxclust")
    else:
        labels = fcluster(z, t=threshold, criterion="distance")
        if len(set(labels)) > max_speakers:
            labels = fcluster(z, t=max_speakers, criterion="maxclust")
    return [int(x) for x in labels]


def relabel_by_first_appearance(labels: list[int]) -> list[str]:
    """Map arbitrary cluster ids to `SPEAKER_00`, `SPEAKER_01`, ... in order of first appearance."""
    seen: dict[int, str] = {}
    out = []
    for lab in labels:
        if lab not in seen:
            seen[lab] = f"SPEAKER_{len(seen):02d}"
        out.append(seen[lab])
    return out


def assign_by_overlap(segments: list["Segment"], turns: list[Turn]) -> list[str]:
    """Assign each segment the speaker of the turn it overlaps most with.

    A segment with no overlap gets the speaker of the nearest turn (by gap).
    """
    out = []
    for seg in segments:
        best_speaker = None
        best_overlap = 0.0
        for turn in turns:
            overlap = min(seg.end, turn.end) - max(seg.start, turn.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker
        if best_speaker is None:
            best_gap = None
            for turn in turns:
                if seg.end <= turn.start:
                    gap = turn.start - seg.end
                elif seg.start >= turn.end:
                    gap = seg.start - turn.end
                else:
                    gap = 0.0
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_speaker = turn.speaker
        out.append(best_speaker)
    return out


def _crop_bounds(start_s: float, end_s: float, sr: int, n_samples: int, min_seconds: float = 0.5) -> tuple[int, int]:
    """Sample-index crop window for a sentence, clamped to `[0, n_samples]` and padded to
    `min_seconds` where the file is long enough to hold it.

    Clamps `[start_s, end_s]` to file bounds first, then grows the (already-clamped) window
    up to `min_seconds`, pushing any padding that would fall outside `[0, n_samples]` on one
    side over to the other side. The result is always within bounds and is never shorter than
    the input segment.
    """
    min_samples = int(round(min_seconds * sr))
    s0 = max(0, min(n_samples, int(round(start_s * sr))))
    s1 = max(0, min(n_samples, int(round(end_s * sr))))
    if s1 < s0:
        s1 = s0

    deficit = min_samples - (s1 - s0)
    if deficit > 0:
        left = deficit // 2
        right = deficit - left
        s0 -= left
        s1 += right
        if s0 < 0:
            s1 += -s0
            s0 = 0
        if s1 > n_samples:
            s0 -= s1 - n_samples
            s1 = n_samples
        s0 = max(0, s0)
        s1 = min(n_samples, s1)

    return s0, s1


def _load_ecapa_embedder(model: str, device: str) -> Callable[[np.ndarray, int, float], np.ndarray]:
    """Return `embed(wav_crop, sr, t0) -> embedding` backed by a speechbrain ECAPA encoder."""
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    from .devices import pick_device

    dev = pick_device(device)
    clf = EncoderClassifier.from_hparams(source=model, run_opts={"device": dev})

    def embed(wav: np.ndarray, sr: int, t0: float) -> np.ndarray:
        x = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0).to(dev)
        with torch.no_grad():
            e = clf.encode_batch(x).squeeze().detach().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    return embed


def _label_ecapa(
    audio_path: str,
    segments: list["Segment"],
    *,
    num_speakers: int | None,
    max_speakers: int,
    threshold: float,
    model: str,
    device: str,
) -> list[str]:
    import soundfile as sf

    wav, sr = sf.read(audio_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    n_samples = len(wav)

    embed = _load_ecapa_embedder(model, device)

    valid_idx = []
    embs = []
    for i, seg in enumerate(segments):
        if seg.end - seg.start < 0.3:
            continue
        s0, s1 = _crop_bounds(seg.start, seg.end, sr, n_samples)
        if s1 <= s0:
            continue
        try:
            e = embed(wav[s0:s1], sr, seg.start)
        except Exception:
            continue
        valid_idx.append(i)
        embs.append(e)

    labels: list[str | None] = [None] * len(segments)
    if embs:
        clustered = cluster_embeddings(
            np.array(embs), num_speakers=num_speakers, max_speakers=max_speakers, threshold=threshold
        )
        for idx, lab in zip(valid_idx, relabel_by_first_appearance(clustered)):
            labels[idx] = lab

    prev = "SPEAKER_00"
    for i, lab in enumerate(labels):
        if lab is None:
            labels[i] = prev
        else:
            prev = lab
    return labels  # type: ignore[return-value]


def _label_pyannote(
    audio_path: str,
    segments: list["Segment"],
    *,
    pyannote_model: str,
    hf_token: str | None,
    device: str,
) -> list[str]:
    from pyannote.audio import Pipeline

    from .devices import pick_device

    pipeline = Pipeline.from_pretrained(pyannote_model, token=hf_token)
    try:
        import torch

        pipeline.to(torch.device(pick_device(device)))
    except Exception:
        pass

    diarization = pipeline(audio_path)
    turns = [Turn(turn.start, turn.end, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]
    return assign_by_overlap(segments, turns)


def label_segments(
    audio_path: str,
    segments: list["Segment"],
    *,
    backend: str = "ecapa",
    num_speakers: int | None = None,
    max_speakers: int = 4,
    threshold: float = 0.65,
    hf_token: str | None = None,
    model: str = "speechbrain/spkrec-ecapa-voxceleb",
    pyannote_model: str = "pyannote/speaker-diarization-community-1",
    device: str = "auto",
) -> list[str]:
    """Return one `SPEAKER_00..` label per segment, in order of first appearance."""
    if not segments:
        return []
    if backend == "pyannote":
        return _label_pyannote(audio_path, segments, pyannote_model=pyannote_model, hf_token=hf_token, device=device)
    return _label_ecapa(
        audio_path,
        segments,
        num_speakers=num_speakers,
        max_speakers=max_speakers,
        threshold=threshold,
        model=model,
        device=device,
    )
