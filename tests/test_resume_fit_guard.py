import json

from resume_from_segments import _check_fit_stage, main


def test_fit_enabled_transcribed_stage_blocked():
    cfg = {"fit": {"enabled": True}}
    msg = _check_fit_stage(cfg, "transcribed")
    assert msg is not None
    assert "fit" in msg.lower()


def test_fit_enabled_fitted_stage_allowed():
    cfg = {"fit": {"enabled": True}}
    assert _check_fit_stage(cfg, "fitted") is None


def test_fit_disabled_transcribed_stage_allowed():
    cfg = {"fit": {"enabled": False}}
    assert _check_fit_stage(cfg, "transcribed") is None


def test_diarized_stage_rejected_regardless_of_fit_enabled():
    # "diarized"/"sentences" checkpoints still hold source-language text —
    # resuming from them must be rejected even when fit.enabled is False.
    msg = _check_fit_stage({"fit": {"enabled": False}}, "diarized")
    assert msg is not None and "diarized" in msg
    msg = _check_fit_stage({"fit": {"enabled": True}}, "sentences")
    assert msg is not None and "sentences" in msg


def test_resume_from_diarized_segments_exits_2(tmp_path, monkeypatch, capsys):
    segments_path = tmp_path / "x.diarized.segments.json"
    segments_path.write_text(
        json.dumps({
            "stage": "diarized",
            "count": 1,
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hello"}],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "resume_from_segments.py",
            "--input", str(tmp_path / "nonexistent.mp4"),
            "--segments", str(segments_path),
            "--output", str(tmp_path / "out.mp4"),
            "--language", "Vietnamese",
            "--config", "config/default.yaml",
        ],
    )
    rc = main()
    assert rc == 2
    assert "diarized" in capsys.readouterr().err
