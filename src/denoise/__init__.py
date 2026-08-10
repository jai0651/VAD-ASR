"""
Module 5: single-channel noise suppression (the stage in front of everything).

`spectral.py` is the from-scratch engine — classical statistical enhancement
(MCRA noise tracking + decision-directed SNR + log-MMSE/OM-LSA gains), which is
what every phone, headset and conferencing stack ran before deep learning and
is still the baseline every neural denoiser is measured against.

`onnx_engines.py` holds the production alternates (GTCRN, DTLN), ported from
the sibling Denoiser project.
"""
