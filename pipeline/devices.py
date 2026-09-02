"""Device selection + memory release shared by local model backends."""

from __future__ import annotations

import gc


def pick_device(requested: str = "auto") -> str:
    """Return 'cuda' | 'mps' | 'cpu'. Explicit values pass through unchanged."""
    if requested and requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def free_memory() -> None:
    """Best-effort release of accelerator memory between pipeline stages."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
