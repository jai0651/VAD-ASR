"""Compatibility shims that must be applied before importing DeepFilterNet.

DeepFilterNet 0.5.6 (the last PyPI release) imports
``torchaudio.backend.common.AudioMetaData``, which newer torchaudio (>=2.1)
removed. We only use DeepFilterNet's tensor-in/tensor-out enhancement path in
this project (never its file I/O), so injecting a stub module is enough to make
the import succeed without affecting behaviour.

Import this module *before* importing anything from ``df``.
"""
from __future__ import annotations

import sys
import types


def apply() -> None:
    if "torchaudio.backend.common" in sys.modules:
        return
    import torchaudio

    if hasattr(torchaudio, "AudioMetaData"):
        return

    class AudioMetaData:  # minimal stand-in; only referenced for typing/imports
        def __init__(self, *args, **kwargs):
            pass

    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = AudioMetaData
    backend.common = common
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common
    torchaudio.backend = backend


apply()
