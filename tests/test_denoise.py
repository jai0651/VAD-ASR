"""
Module 5 tests: the properties a streaming denoiser must have, independent of
which engine is selected.

The interesting ones are 2 and 4. (2) is the perfect-reconstruction check: with
the gain floor raised to 0 dB the whole algorithm collapses to gain = 1, so the
output must equal the input — which tests the sqrt-Hann/COLA overlap-add
skeleton on its own. If that fails, every quality number downstream is
measuring a broken STFT rather than a denoiser. (4) is the alignment check that
`denoise_target=vad` depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.denoise.spectral import HOP, SpectralDenoiser
from src.pipeline.denoise import DenoisingVAD, PassthroughDenoiser
from src.pipeline.vad import VADResult

SR = 16_000


def _speech_like(n: int, seed: int = 0) -> np.ndarray:
    """A crude voiced-speech surrogate: a 120 Hz pitch + harmonics, amplitude
    modulated at syllable rate. Not real speech, but it has the property that
    matters here — energy concentrated in a few bands, with pauses."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    sig = sum(np.sin(2 * np.pi * 120 * k * t) / k for k in range(1, 12))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t + rng.uniform(0, 6))
    return (0.2 * sig * envelope).astype(np.float32)


def _real_speech(seconds: float = 4.0) -> np.ndarray:
    """A real LibriSpeech utterance. The SNR tests MUST use real speech: a
    trained denoiser recognises speech by its learned time-frequency shape, so
    it correctly suppresses the synthetic buzz above as noise. Testing a neural
    denoiser on a synthetic signal measures nothing."""
    import random

    from src.vad.data_real import list_speech_files, load_clip

    try:
        files = list_speech_files()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    random.seed(0)  # load_clip takes a RANDOM crop; pin it so results repeat
    return load_clip(sorted(files)[0], max_s=seconds)


def _stream(engine, x: np.ndarray, block: int = 320) -> np.ndarray:
    out = [engine.process(x[i:i + block]) for i in range(0, len(x), block)]
    return np.concatenate([o for o in out if len(o)]) if out else np.zeros(0, np.float32)


def _snr_db(clean: np.ndarray, mixed: np.ndarray) -> float:
    n = min(len(clean), len(mixed))
    noise = mixed[:n] - clean[:n]
    return 10 * np.log10((np.sum(clean[:n] ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12))


def _mix_at_snr(speech: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Mix white noise in at a *specified* input SNR. Fixing the noise
    amplitude instead leaves the input SNR at the mercy of the clip's loudness;
    at 19 dB in there is nothing left to remove and every denoiser 'loses'."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(speech)).astype(np.float32)
    scale = np.sqrt(np.mean(speech ** 2)) / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
    return speech + noise * scale * (10.0 ** (-snr_db / 20.0))


def test_passthrough_is_identity():
    d = PassthroughDenoiser()
    x = _speech_like(8000)
    assert np.array_equal(_stream(d, x), x)


def test_zero_attenuation_reconstructs_perfectly():
    """max_atten_db=0 pins the gain to 1.0, so this measures ONLY the STFT ->
    overlap-add -> ISTFT skeleton. Periodic sqrt-Hann at 50% overlap sums to
    exactly 1, so reconstruction should be near machine precision."""
    d = SpectralDenoiser(max_atten_db=0.0)
    x = _speech_like(16000)
    y = _stream(d, x)
    # Output lags input by one hop; compare the overlapping region.
    n = len(y) - HOP
    assert n > 8000
    np.testing.assert_allclose(y[HOP:HOP + n], x[:n], atol=2e-5)


def test_suppresses_stationary_noise():
    rng = np.random.default_rng(7)
    noise = (0.05 * rng.standard_normal(SR * 2)).astype(np.float32)
    d = SpectralDenoiser(max_atten_db=18.0)
    y = _stream(d, noise)
    in_rms = np.sqrt(np.mean(noise[HOP:len(y)] ** 2))
    out_rms = np.sqrt(np.mean(y[HOP:] ** 2))
    reduction_db = 20 * np.log10(in_rms / (out_rms + 1e-12))
    assert reduction_db > 10.0, f"only {reduction_db:.1f} dB of suppression"


def test_improves_snr_on_noisy_speech():
    speech = _real_speech(4.0)
    mixed = _mix_at_snr(speech, snr_db=5.0, seed=3)

    d = SpectralDenoiser(max_atten_db=18.0)
    y = _stream(d, mixed)
    n = len(y) - HOP

    before = _snr_db(speech[:n], mixed[:n])
    after = _snr_db(speech[:n], y[HOP:HOP + n])
    # Deliberately a low bar. Classical enhancement buys only ~1-2 dB of SNR
    # because it also attenuates the speech (~3 dB) while it removes noise
    # (~15 dB in pauses). GTCRN buys ~7 dB on the same mixture with ~1 dB of
    # speech attenuation — that gap IS the value of a trained model, and
    # scripts/10_denoise_bench.py reports it.
    assert after > before + 1.0, f"SNR {before:.1f} -> {after:.1f} dB"


def test_attenuates_pauses_much_more_than_speech():
    """The defining property of a denoiser, and the one that survives across
    engines: silence should get quiet, speech should not."""
    speech = _real_speech(4.0)
    mixed = _mix_at_snr(speech, snr_db=5.0, seed=3)
    y = _stream(SpectralDenoiser(max_atten_db=18.0), mixed)
    n = len(y) - HOP
    clean, out, noisy = speech[:n], y[HOP:HOP + n], mixed[:n]

    env = np.convolve(clean ** 2, np.ones(400) / 400, mode="same")
    is_speech = env > 0.02 * env.max()

    def atten_db(mask):
        a = np.sqrt(np.mean(noisy[mask] ** 2)) + 1e-12
        b = np.sqrt(np.mean(out[mask] ** 2)) + 1e-12
        return 20 * np.log10(a / b)

    in_pauses, in_speech = atten_db(~is_speech), atten_db(is_speech)
    assert in_pauses > 10.0, f"pauses only attenuated {in_pauses:.1f} dB"
    assert in_speech < 6.0, f"speech attenuated {in_speech:.1f} dB — eating the signal"


def test_block_size_does_not_change_output():
    """A streaming engine must be block-size agnostic: the same audio fed in
    320-sample or 1000-sample chunks has to produce identical samples. This is
    what makes the WebSocket frame size a free parameter."""
    x = _speech_like(SR)
    a = _stream(SpectralDenoiser(), x, block=320)
    b = _stream(SpectralDenoiser(), x, block=1000)
    n = min(len(a), len(b))
    np.testing.assert_allclose(a[:n], b[:n], atol=1e-6)


def test_reset_restores_initial_state():
    d = SpectralDenoiser()
    x = _speech_like(SR)
    a = _stream(d, x)
    d.reset()
    b = _stream(d, x)
    np.testing.assert_allclose(a, b, atol=1e-6)


# --------------------------------------------------------------------------
# pipeline integration
# --------------------------------------------------------------------------
class _FakeVAD:
    """Minimal VAD engine: emits one result per `window` samples, prob = RMS."""

    window = 160

    def __init__(self):
        self._buf = np.zeros(0, np.float32)

    def push(self, samples):
        self._buf = np.concatenate([self._buf, samples])
        out = []
        while len(self._buf) >= self.window:
            win, self._buf = self._buf[: self.window], self._buf[self.window:]
            out.append(VADResult(prob=float(np.sqrt(np.mean(win ** 2))), samples=win))
        return out

    def reset(self):
        self._buf = np.zeros(0, np.float32)


def test_target_vad_hands_downstream_the_aligned_original():
    """denoise_target='vad': the VAD scores cleaned audio but the endpointer
    (and therefore the ASR) must receive the ORIGINAL samples, aligned to the
    same instants. Misalignment here clips word onsets."""
    x = _speech_like(SR)
    stage = DenoisingVAD(_FakeVAD(), SpectralDenoiser(), pass_clean_downstream=False)

    got = []
    for i in range(0, len(x), 320):
        got.extend(r.samples for r in stage.push(x[i:i + 320]))
    got = np.concatenate(got)

    delay = stage.denoiser.delay
    expected = np.concatenate([np.zeros(delay, np.float32), x])[: len(got)]
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_target_both_hands_downstream_the_cleaned_audio():
    rng = np.random.default_rng(11)
    x = (0.05 * rng.standard_normal(SR)).astype(np.float32)  # pure noise
    stage = DenoisingVAD(_FakeVAD(), SpectralDenoiser(), pass_clean_downstream=True)
    got = np.concatenate([
        r.samples for i in range(0, len(x), 320) for r in stage.push(x[i:i + 320])
    ])
    assert np.sqrt(np.mean(got ** 2)) < 0.5 * np.sqrt(np.mean(x ** 2))
    assert stage.reduction_db > 6.0


def _make_engine(name: str):
    pytest.importorskip("onnxruntime")
    from src.denoise.onnx_engines import DTLNDenoiser, GTCRNDenoiser

    try:
        return {"gtcrn": GTCRNDenoiser, "dtln": DTLNDenoiser}[name]()
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize("name", ["spectral", "gtcrn", "dtln"])
def test_declared_delay_matches_measured_delay(name):
    """`delay` is what keeps the raw and cleaned streams aligned in
    denoise_target='vad'. Measure it instead of trusting it: send an impulse
    through with suppression effectively disabled and find where it lands.
    (DTLN's is 384, not one hop — rectangular 75%-overlap framing.)"""
    if name == "spectral":
        d = SpectralDenoiser(max_atten_db=0.0)  # gain pinned to 1 -> pure shell
    else:
        d = _make_engine(name)
        d._resid = 1.0  # residual mix at unity -> bypasses the model, keeps the shell

    x = np.zeros(8000, np.float32)
    x[2000] = 1.0
    y = _stream(d, x, block=320)
    assert int(np.argmax(np.abs(y))) - 2000 == d.delay


@pytest.mark.parametrize("engine", ["gtcrn", "dtln"])
def test_onnx_engines_suppress_noise(engine):
    onnx = pytest.importorskip("onnxruntime")  # noqa: F841
    from src.denoise.onnx_engines import DTLNDenoiser, GTCRNDenoiser

    try:
        d = GTCRNDenoiser() if engine == "gtcrn" else DTLNDenoiser()
    except FileNotFoundError as e:
        pytest.skip(str(e))

    speech = _real_speech(4.0)
    mixed = _mix_at_snr(speech, snr_db=5.0, seed=5)
    y = _stream(d, mixed)
    n = len(y) - d.delay

    before = _snr_db(speech[:n], mixed[:n])
    after = _snr_db(speech[:n], y[d.delay:d.delay + n])
    assert after > before + 2.0, f"{engine}: SNR {before:.1f} -> {after:.1f} dB"
