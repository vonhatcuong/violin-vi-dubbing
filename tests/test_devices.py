from pipeline import devices


def test_explicit_device_is_returned_verbatim():
    assert devices.pick_device("cpu") == "cpu"
    assert devices.pick_device("cuda") == "cuda"


def test_auto_picks_a_known_device():
    assert devices.pick_device("auto") in {"cpu", "mps", "cuda"}


def test_free_memory_never_raises():
    devices.free_memory()
