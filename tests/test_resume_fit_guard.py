from resume_from_segments import _check_fit_stage


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
