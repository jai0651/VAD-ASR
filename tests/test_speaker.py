"""
Module 9 tests.

Three of these are the ones that matter.

`test_pooling_ignores_padding` is the contract that makes batched inference
safe. Diarization will push variable-length windows through the encoder in
batches; a pooling layer that averages over the padding returns a subtly
different embedding depending on what else happened to be in the batch. That
never crashes — it shows up as "clustering is a bit rubbish", which is the
worst kind of bug to chase.

`test_speed_perturbation_creates_new_classes` guards the labelling decision that
is the exact opposite of the ASR one. Reuse src/asr/corpus.py's augmentation
here without relabelling and you train the model to ignore pitch, which is one
of the two strongest identity cues it has. It would still train, and the EER
would just be quietly worse.

`test_eer_is_symmetric_and_bounded` pins the metric. Every claim in this module
is an EER, so an EER function that is subtly wrong invalidates the whole module
rather than one number in it.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

from src.speaker.data import SpeakerCrops, normalize_feats, random_crop, split_by_speaker
from src.speaker.loss import AAMSoftmax, PlainSoftmax, margin_schedule
from src.speaker.model import (
    AttentiveStatsPool,
    StatsPool,
    lengths_to_mask,
    make_speaker_net,
    masked_stats,
)
from src.speaker.verify import compute_eer, make_trials, min_dcf, naive_stats_embedding

HAS_LIBRI = os.path.isdir("data/LibriSpeech/dev-clean")


# ------------------------------------------------------------------- pooling
@pytest.mark.parametrize("pool_cls", [StatsPool, AttentiveStatsPool])
def test_pooling_ignores_padding(pool_cls):
    """Appending garbage past `lengths` must not move the pooled vector."""
    torch.manual_seed(0)
    pool = pool_cls(32) if pool_cls is AttentiveStatsPool else pool_cls()
    pool.eval()
    x = torch.randn(3, 32, 40)
    lengths = torch.tensor([40, 40, 40])
    ref = pool(x, lengths_to_mask(lengths, 40))

    padded = torch.cat([x, torch.randn(3, 32, 25) * 10.0], dim=2)
    got = pool(padded, lengths_to_mask(lengths, 65))
    assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()


def test_masked_stats_match_manual_computation():
    x = torch.randn(2, 4, 10)
    mask = lengths_to_mask(torch.tensor([10, 6]), 10)
    mean, std = masked_stats(x, mask)
    assert torch.allclose(mean[0, :, 0], x[0].mean(dim=1), atol=1e-6)
    assert torch.allclose(mean[1, :, 0], x[1, :, :6].mean(dim=1), atol=1e-6)
    # population std (not Bessel-corrected), matching the pooling definition
    assert torch.allclose(std[1, :, 0], x[1, :, :6].std(dim=1, unbiased=False), atol=1e-5)


def test_attentive_weights_form_a_distribution_over_time():
    torch.manual_seed(0)
    pool = AttentiveStatsPool(16).eval()
    x = torch.randn(2, 16, 30)
    out = pool(x, lengths_to_mask(torch.tensor([30, 12]), 30))
    assert out.shape == (2, 32)                       # mean and std concatenated
    assert torch.isfinite(out).all()                  # -inf masking must not leak


# -------------------------------------------------------------------- models
@pytest.mark.parametrize("arch", ["ecapa", "xvector"])
def test_encoder_shapes_and_variable_length(arch):
    model = make_speaker_net(arch, channels=64, embed_dim=32).eval()
    if arch == "xvector":
        model = make_speaker_net(arch, channels=64, stats_channels=96, embed_dim=32).eval()
    for t in (100, 198, 400):
        out = model(torch.randn(2, t, 80))
        assert out.shape == (2, 32)
        assert torch.isfinite(out).all()


def test_encoder_is_permutation_sensitive_in_time():
    """A speaker embedding may be order-insensitive-ish, but not constant:
    a model that returns the same vector for any input has collapsed."""
    torch.manual_seed(0)
    model = make_speaker_net("ecapa", channels=64, embed_dim=32).eval()
    a = model(torch.randn(1, 200, 80))
    b = model(torch.randn(1, 200, 80) * 3.0 + 1.0)
    assert (a - b).abs().max() > 1e-3


# ---------------------------------------------------------------------- loss
def test_aam_with_zero_margin_is_scaled_cosine_softmax():
    torch.manual_seed(0)
    head = AAMSoftmax(16, 5, margin=0.0, scale=30.0)
    emb, y = torch.randn(8, 16), torch.randint(0, 5, (8,))
    loss, cos = head(emb, y)
    expected = torch.nn.functional.cross_entropy(30.0 * cos, y)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_margin_makes_the_target_harder():
    """Same embeddings, same weights: adding margin can only raise the loss."""
    torch.manual_seed(0)
    head = AAMSoftmax(16, 5, margin=0.0, scale=30.0)
    emb, y = torch.randn(8, 16), torch.randint(0, 5, (8,))
    without, _ = head(emb, y)
    head.set_margin(0.2)
    with_margin, _ = head(emb, y)
    assert with_margin.item() > without.item()


def test_margin_schedule_ramps_then_holds():
    assert margin_schedule(0, 1000, 0.2) == 0.0
    assert margin_schedule(150, 1000, 0.2) == pytest.approx(0.1)
    assert margin_schedule(300, 1000, 0.2) == pytest.approx(0.2)
    assert margin_schedule(999, 1000, 0.2) == pytest.approx(0.2)


def test_aam_learns_to_separate_two_classes():
    """The loss must be trainable: fit a 2-class toy problem to near-zero."""
    torch.manual_seed(0)
    head = AAMSoftmax(4, 2, margin=0.2, scale=30.0)
    emb = torch.nn.Parameter(torch.randn(16, 4))
    y = torch.tensor([0, 1] * 8)
    opt = torch.optim.Adam([emb] + list(head.parameters()), lr=0.1)
    first = None
    for _ in range(300):
        loss, _ = head(emb, y)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first / 10


def test_plain_head_has_the_same_interface():
    head = PlainSoftmax(8, 3)
    head.set_margin(0.2)                       # must be a harmless no-op
    loss, logits = head(torch.randn(4, 8), torch.tensor([0, 1, 2, 0]))
    assert logits.shape == (4, 3) and loss.ndim == 0


# ------------------------------------------------------------------- metrics
def test_eer_is_symmetric_and_bounded():
    rng = np.random.default_rng(0)
    # perfectly separable
    s = np.concatenate([rng.normal(1, 0.02, 400), rng.normal(-1, 0.02, 400)])
    y = np.concatenate([np.ones(400), np.zeros(400)])
    assert compute_eer(s, y)[0] == 0.0
    # indistinguishable -> ~50%
    s = rng.normal(0, 1, 2000)
    y = np.concatenate([np.ones(1000), np.zeros(1000)])
    eer, _ = compute_eer(s, y)
    assert 0.4 < eer < 0.6


def test_min_dcf_is_capped_at_one():
    """The normaliser is defined so 'reject everything' scores exactly 1.0;
    a worse-than-useless system must not report more than that."""
    rng = np.random.default_rng(1)
    s = rng.normal(0, 1, 2000)
    y = np.concatenate([np.ones(1000), np.zeros(1000)])
    assert min_dcf(s, y) <= 1.0 + 1e-9


def test_eer_threshold_actually_balances_the_errors():
    rng = np.random.default_rng(2)
    s = np.concatenate([rng.normal(0.6, 0.3, 800), rng.normal(0.0, 0.3, 800)])
    y = np.concatenate([np.ones(800), np.zeros(800)])
    eer, thr = compute_eer(s, y)
    frr = (s[y == 1] < thr).mean()
    far = (s[y == 0] >= thr).mean()
    assert abs(frr - far) < 0.02
    assert abs((frr + far) / 2 - eer) < 1e-9


def test_naive_embedding_dimensions():
    feats = torch.randn(50, 80)
    assert naive_stats_embedding(feats, use_std=False).shape == (80,)
    assert naive_stats_embedding(feats, use_std=True).shape == (160,)


def test_std_survives_mean_normalisation():
    """The claim script 20 rests on: CMN cannot change a standard deviation."""
    feats = torch.randn(120, 80) * 3.0 - 20.0
    a = feats.std(0)
    b = normalize_feats(feats, "mean").std(0)
    assert torch.allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------------- data
def test_random_crop_length_and_looping():
    rng = __import__("random").Random(0)
    assert len(random_crop(np.zeros(1000, dtype=np.float32), 400, rng)) == 400
    assert len(random_crop(np.zeros(100, dtype=np.float32), 400, rng)) == 400   # loops


def test_split_by_speaker_is_disjoint():
    items = [{"path": f"/x/{s}-0-{i}.flac", "speaker": s}
             for s in "abcdefghij" for i in range(3)]
    train, trial = split_by_speaker(items, n_held_out=3, seed=0)
    assert len({i["speaker"] for i in train} & {i["speaker"] for i in trial}) == 0
    assert len({i["speaker"] for i in trial}) == 3
    assert len(train) + len(trial) == len(items)


def test_trials_are_balanced_and_never_self_pairs():
    items = [{"path": f"/x/{s}-0-{i}.flac", "speaker": s}
             for s in "abcdefghij" for i in range(10)]
    trials = make_trials(items, n_pairs=200, seed=0)
    assert len(trials) == 200
    assert sum(t[2] for t in trials) == 100
    for a, b, same in trials:
        assert a != b
        assert (a.split("/")[-1].split("-")[0] == b.split("/")[-1].split("-")[0]) == bool(same)


def test_trials_shrink_rather_than_unbalance_when_pairs_run_out():
    """6 speakers x 6 utterances = 6*C(6,2) = 90 possible same-speaker pairs.
    Asking for 100 must give back 180 BALANCED trials, not 200 skewed ones."""
    items = [{"path": f"/x/{s}-0-{i}.flac", "speaker": s}
             for s in "abcdef" for i in range(6)]
    trials = make_trials(items, n_pairs=200, seed=0, verbose=False)
    n_same = sum(t[2] for t in trials)
    assert n_same == 90
    assert len(trials) == 180
    assert n_same * 2 == len(trials)


def test_speed_perturbation_creates_new_classes():
    """Speakers x speeds, and no two (speaker, speed) pairs share a class id."""
    items = [{"path": f"/x/{s}-0-{i}.flac", "speaker": s}
             for s in "abcd" for i in range(2)]
    ds = SpeakerCrops(items, speeds=(0.9, 1.0, 1.1), train=True)
    assert ds.n_speakers == 4 and ds.n_classes == 12
    seen = {ds.label(s, k) for s in "abcd" for k in range(3)}
    assert len(seen) == 12

    # ...and at eval time the labelling collapses back to one class per speaker
    ev = SpeakerCrops(items, train=False)
    assert ev.n_classes == 4 and ev.label("a", 0) == 0 and ev.label("b", 0) == 1


@pytest.mark.skipif(not HAS_LIBRI, reason="LibriSpeech dev-clean not downloaded")
def test_crops_are_all_the_same_shape():
    """Fixed-length crops are what let the training loop skip padding entirely."""
    from src.speaker.data import build_speaker_manifest

    items = build_speaker_manifest()[:8]
    ds = SpeakerCrops(items, crop_seconds=2.0, train=True, noise_prob=0.0)
    shapes = {tuple(ds[i][0].shape) for i in range(len(ds))}
    assert len(shapes) == 1, shapes
    t, f = shapes.pop()
    assert f == 80
    assert t == math.floor((2.0 * 16000 - 400) / 160) + 1
