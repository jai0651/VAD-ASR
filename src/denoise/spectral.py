"""
Module 5, part 1: a streaming noise suppressor built from scratch.

Everything upstream in this repo assumed the mic was reasonably clean. It never
is: fans, keyboards, traffic, a TV in the next room, and — the one that actually
broke us in Module 1b — room ambience that a VAD reads as speech. Noise
suppression is the stage that makes every later stage's job easier, and it is
the one place in a voice pipeline where a *non-neural* algorithm is still
genuinely competitive, which makes it a great thing to write by hand.

THE MODEL OF THE WORLD.  We assume additive noise in the STFT domain:

    Y[k,t] = S[k,t] + N[k,t]                (noisy = speech + noise, per bin)

and we estimate a real, non-negative gain G[k,t] in [Gmin, 1] to apply to each
bin:  Ŝ = G · Y.  Phase is left untouched — a huge simplification, and the exact
place where modern complex-domain models (GTCRN) beat this one.

Everything reduces to three questions, each answered by one classic algorithm:

  1. HOW LOUD IS THE NOISE, per bin, right now?  We can't ask the user to be
     quiet, so we track it *during speech* with MCRA (Minima-Controlled
     Recursive Averaging, Cohen & Berdugo 2002): speech is intermittent, so the
     running MINIMUM of the smoothed power in each bin is a good noise floor.
     Where the current power sits far above that minimum we're probably in
     speech, so we freeze the noise estimate; where it doesn't, we update it.
     Minimum tracking uses Doblinger's continuous update — no O(window) history.

  2. HOW MUCH SPEECH IS IN THIS BIN?  The a-priori SNR ξ. Estimating it from the
     current frame alone is horribly noisy and produces "musical noise" (random
     isolated tone bursts — the classic spectral-subtraction artifact). Ephraim
     & Malah's DECISION-DIRECTED estimator fixes it by averaging the previous
     frame's *enhanced* SNR with this frame's raw estimate. That temporal
     smoothing is the single most important line in this file.

  3. GIVEN ξ, WHAT GAIN IS OPTIMAL?  Minimising squared error on the *log*
     spectrum (log-MMSE, Ephraim & Malah 1985) rather than the linear one,
     because loudness perception is logarithmic. That yields a gain built from
     the exponential integral E1 — implemented here with the Abramowitz &
     Stegun rational approximations, since we have no scipy.

  4. …and one refinement: OM-LSA (Cohen 2003) blends the log-MMSE gain toward
     the floor using the speech-presence probability from step 1, so bins that
     are confidently noise-only get attenuated hard and speech bins are left
     alone. This is what kills the residual musical noise.

STREAMING CONTRACT (shared by every engine in this module): feed consecutive,
non-overlapping blocks in stream order; `process` buffers internally to `hop`
and returns only fully reconstructed hops, so it may return fewer samples than
you gave it. Output sample i still corresponds to input sample i — it just
arrives one hop later. That one-hop delay IS the algorithmic latency.
"""

from __future__ import annotations

import numpy as np

SR = 16_000
NFFT = 512      # 32 ms analysis window — long enough to resolve pitch harmonics
HOP = 256       # 16 ms hop = 50% overlap = the output granularity / added latency

# Periodic (not symmetric) Hann. Applied as sqrt on analysis AND synthesis, so
# the two multiply back to a full Hann, and a full periodic Hann summed at 50%
# overlap is exactly 1.0 — perfect reconstruction with no normalisation pass.
_n = np.arange(NFFT)
HANN = 0.5 - 0.5 * np.cos(2.0 * np.pi * _n / NFFT)
WINDOW = np.sqrt(HANN).astype(np.float32)

EPS = 1e-12


# ---------------------------------------------------------------------------
# The exponential integral E1(x) = ∫_x^∞ e^-t / t dt
# ---------------------------------------------------------------------------
def _expint1(x: np.ndarray) -> np.ndarray:
    """E1 via Abramowitz & Stegun 5.1.53 / 5.1.56 (no scipy needed).

    Two regimes, because no single cheap polynomial covers both:
      x <= 1 : a 5-term series plus the -ln(x) singularity  (|err| < 2e-7)
      x >  1 : a rational approximation to x·e^x·E1(x)      (|err| < 5e-5)
    Accuracy far beyond what a gain curve needs; the point is that it's exact
    enough that the log-MMSE gain is the real thing, not an approximation of it.
    """
    x = np.maximum(x, 1e-8)
    small = x <= 1.0

    a = (-0.57721566, 0.99999193, -0.24991055, 0.05519968, -0.00976004, 0.00107857)
    xs = np.where(small, x, 1.0)  # dummy value where unused, to avoid log(big)
    ser = a[0] + xs * (a[1] + xs * (a[2] + xs * (a[3] + xs * (a[4] + xs * a[5]))))
    lo = -np.log(xs) + ser

    xl = np.where(small, 2.0, x)  # dummy where unused
    num = xl * xl + 2.334733 * xl + 0.250621
    den = xl * xl + 3.330657 * xl + 1.681534
    hi = np.exp(-xl) / xl * (num / den)

    return np.where(small, lo, hi)


# ---------------------------------------------------------------------------
class SpectralDenoiser:
    """Streaming OM-LSA noise suppressor. ~0 parameters, ~0.5% of a CPU core.

    Parameters
    ----------
    max_atten_db : the gain floor, i.e. how hard a noise-only bin may be
        attenuated. Full suppression sounds "cleaner" in isolation but leaves
        an unnatural pumping silence AND is measurably worse for ASR, which was
        trained on audio that still has a noise floor. 15-20 dB is the sane
        range for a speech pipeline; Krisp-style consumer NR goes further
        because a human, not an ASR, is listening.
    alpha : decision-directed smoothing. 0.98 is the canonical value; lower
        reacts faster to onsets but brings musical noise back.
    """

    name = "spectral"
    sr = SR
    hop = HOP
    # Overlap-add lag: the hop we emit is the first half of the window we just
    # analysed, so output sample i corresponds to input sample i - HOP.
    delay = HOP

    def __init__(
        self,
        max_atten_db: float = 18.0,
        alpha: float = 0.98,
        noise_warmup_frames: int = 6,
    ):
        self.g_min = float(10.0 ** (-abs(max_atten_db) / 20.0))
        self.alpha = float(alpha)
        self.warmup = int(noise_warmup_frames)
        self.n_bins = NFFT // 2 + 1

        # --- MCRA constants (Cohen & Berdugo 2002 / Doblinger 1995) ---
        self._a_s = 0.8      # smoothing of the noisy power (S)
        self._gamma = 0.998  # Doblinger minimum-tracking decay
        self._beta = 0.96    # Doblinger look-back weight
        # delta / b_min are the two that actually move the numbers, and the
        # textbook values are tuned for 10 ms hops on long stationary noise.
        # Swept over 6 LibriSpeech clips at 5 dB SNR (scripts/10 protocol):
        # delta 5.0 -> +3.8 dB SNR with 2.6 dB of speech attenuation;
        # delta 2.0 -> +5.9 dB SNR with 1.7 dB. A lower threshold declares
        # speech-presence more readily, which stops the noise estimate from
        # absorbing speech in the low bands where it is nearly continuous.
        self._delta = 2.0    # S/Smin ratio above which we call it speech
        self._a_p = 0.2      # smoothing of the speech-presence indicator
        self._a_d = 0.95     # noise-update smoothing when speech is absent
        self._b_min = 1.5    # bias compensation on the tracked minimum

        self.reset()

    # -- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        z = np.zeros(self.n_bins, dtype=np.float64)
        self._S = z.copy()          # smoothed noisy power
        self._S_prev = z.copy()     # previous smoothed power (Doblinger)
        self._S_min = z.copy()      # tracked minimum = noise floor proxy
        self._p = z.copy()          # smoothed speech-presence indicator
        self._lambda_d = z.copy()   # the noise PSD estimate
        self._gain_prev = np.ones(self.n_bins)      # G[k, t-1]
        self._gamma_prev = np.ones(self.n_bins)     # a-posteriori SNR[k, t-1]
        self._frame = 0

        self._in = np.zeros(NFFT, dtype=np.float32)   # sliding analysis frame
        self._ola = np.zeros(NFFT, dtype=np.float32)  # overlap-add accumulator
        self._buf = np.zeros(0, dtype=np.float32)     # not-yet-hopped input
        self._in_rms = 0.0
        self._out_rms = 0.0

    # -- stream -------------------------------------------------------------
    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        self._buf = np.concatenate([self._buf, block])
        out = []
        while self._buf.shape[0] >= HOP:
            out.append(self._process_hop(self._buf[:HOP]))
            self._buf = self._buf[HOP:]
        if not out:
            return np.zeros(0, dtype=np.float32)
        y = np.concatenate(out)
        # Level telemetry: how many dB of energy did we remove? Watching this
        # live is the fastest way to tell "NR is working" from "NR is eating
        # the speech" — a healthy value on quiet speech is a few dB, not 20.
        n = max(1, y.shape[0])
        self._in_rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2) + EPS))
        self._out_rms = float(np.sqrt(np.sum(y.astype(np.float64) ** 2) / n + EPS))
        return y

    @property
    def reduction_db(self) -> float:
        return float(20.0 * np.log10((self._in_rms + EPS) / (self._out_rms + EPS)))

    # -- the algorithm ------------------------------------------------------
    def _process_hop(self, new: np.ndarray) -> np.ndarray:
        # Slide the analysis frame forward by one hop.
        self._in = np.roll(self._in, -HOP)
        self._in[-HOP:] = new

        spec = np.fft.rfft(self._in * WINDOW)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float64)

        gain = self._gain(power)

        frame = np.fft.irfft(spec * gain).astype(np.float32) * WINDOW
        self._ola = np.roll(self._ola, -HOP)
        self._ola[-HOP:] = 0.0
        self._ola += frame
        return self._ola[:HOP].copy()

    def _gain(self, power: np.ndarray) -> np.ndarray:
        self._frame += 1

        # ---- 1. noise PSD: MCRA ------------------------------------------
        if self._frame <= self.warmup:
            # Bootstrap: assume the first few frames are noise-only. Every
            # denoiser makes this assumption; it's why they misbehave if you
            # start talking in the first 100 ms.
            k = self._frame
            self._lambda_d += (power - self._lambda_d) / k
            self._S = self._lambda_d.copy()
            self._S_min = self._lambda_d.copy()
            self._S_prev = self._lambda_d.copy()
        else:
            self._S = self._a_s * self._S + (1 - self._a_s) * power

            # Doblinger continuous minimum tracking: decay toward the current
            # value when we're above the minimum, snap down when below. O(1)
            # per bin, no history buffer — this is why it ships on DSPs.
            below = self._S_min < self._S
            tracked = (
                self._gamma * self._S_min
                + ((1 - self._gamma) / (1 - self._beta))
                * (self._S - self._beta * self._S_prev)
            )
            self._S_min = np.where(below, tracked, self._S)
            self._S_prev = self._S.copy()

            # Speech present where the smoothed power towers over the minimum.
            indicator = (self._S / np.maximum(self._S_min, EPS) > self._delta)
            self._p = self._a_p * self._p + (1 - self._a_p) * indicator
            # Freeze the noise estimate in speech bins (a_d -> 1), update it
            # fast in noise-only bins.
            a_tilde = self._a_d + (1 - self._a_d) * self._p
            self._lambda_d = a_tilde * self._lambda_d + (1 - a_tilde) * power

            # SAFETY NET (Martin's minimum statistics). Recursive averaging
            # alone has a nasty failure mode: if it is ever initialised high —
            # e.g. the stream starts mid-word, which for a voice agent is the
            # NORMAL case — the freeze-during-speech rule keeps it high, and a
            # too-high noise floor over-suppresses the very bands speech lives
            # in. Measured on a LibriSpeech crop starting mid-utterance: +9 dB
            # of bias below 500 Hz. The minimum of the smoothed power cannot be
            # far above the true noise floor, so we cap the estimate by it.
            # B_MIN compensates the downward bias of taking a minimum.
            self._lambda_d = np.minimum(self._lambda_d, self._b_min * self._S_min)

        lam = np.maximum(self._lambda_d, EPS)

        # ---- 2. SNRs -----------------------------------------------------
        gamma_post = np.minimum(power / lam, 1e4)              # a posteriori
        xi = self.alpha * (self._gain_prev ** 2) * self._gamma_prev + (
            1 - self.alpha
        ) * np.maximum(gamma_post - 1.0, 0.0)                   # decision-directed
        xi = np.maximum(xi, 1e-4)                               # -40 dB floor

        # ---- 3. log-MMSE gain --------------------------------------------
        v = np.clip(xi / (1.0 + xi) * gamma_post, 1e-8, 500.0)
        g_lsa = xi / (1.0 + xi) * np.exp(0.5 * _expint1(v))
        g_lsa = np.clip(g_lsa, 0.0, 1.0)

        # ---- 4. OM-LSA: blend toward the floor by presence probability ----
        # The blend exponent must be a real SPEECH-PRESENCE PROBABILITY, not the
        # MCRA indicator `self._p`. They answer different questions: `self._p`
        # asks "is this band above its own long-run minimum?" (good enough to
        # decide whether to update the noise estimate), while the gain needs
        # "given ξ and γ, how likely is speech here *now*?". Cohen's estimator,
        # from the same Gaussian model as the log-MMSE gain:
        #
        #     p = 1 / (1 + q/(1-q) · (1+ξ) · e^-v)
        #
        # with q the a-priori speech-ABSENCE probability, which is where the
        # MCRA indicator earns its keep. Using `self._p` directly instead cost
        # 5 dB of speech attenuation (vs. 2 dB here) — it never saturates to 1
        # inside speech, so every voiced band got dragged toward the floor.
        q = np.clip(1.0 - self._p, 0.02, 0.98)
        spp = 1.0 / (1.0 + (q / (1.0 - q)) * (1.0 + xi) * np.exp(-v))

        # G = G_lsa^p · Gmin^(1-p). p≈1 (speech) -> untouched; p≈0 (noise) ->
        # floored. The exponential blend is smooth in time, which is what
        # stops the "underwater" warble of a hard binary mask.
        gain = (g_lsa ** spp) * (self.g_min ** (1.0 - spp))
        gain = np.clip(gain, self.g_min, 1.0)

        self._gain_prev = gain
        self._gamma_prev = gamma_post
        return gain
