"""
Module 6 tests.

Two of these are the ones that matter.

`test_chunked_encoder_is_truly_causal` is the streaming contract: if a chunked
encoder's output for frame t changes when you append future audio, then every
streaming latency number you report is a lie and the model will behave
differently online than it did in eval. This is the single easiest thing to get
silently wrong in a Conformer (one wrong mask broadcast, one non-causal conv).

`test_overfits_a_tiny_batch` is the sanity test for any seq2seq model: a model
that cannot memorise two utterances has a bug, not a data problem. Running it
before a multi-hour training job has saved more time than any other test here.
"""

from __future__ import annotations

import torch

from src.asr.conformer import ConformerEncoder, make_chunk_mask, make_pad_mask
from src.asr.hybrid import HybridCTCAttention, count_parameters
from src.asr.search import corpus_wer, ctc_prefix_beam_search, word_error_rate
from src.asr.tokenizer import BLANK_ID, SOS_ID, BPETokenizer, normalize

TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "she sells sea shells by the sea shore",
    "a recognition system must recognise recognisable speech",
    "don't stop believing hold on to that feeling",
] * 12


# --------------------------------------------------------------------- tokenizer
def test_bpe_roundtrip_is_lossless():
    tok = BPETokenizer.train(TEXTS, vocab_size=200)
    for t in TEXTS[:4]:
        assert tok.decode(tok.encode(t)) == normalize(t)


def test_bpe_learns_multichar_units():
    tok = BPETokenizer.train(TEXTS, vocab_size=200)
    ids = tok.encode("the quick brown fox")
    # If BPE worked, 19 characters compress to noticeably fewer tokens.
    assert len(ids) < 19
    assert any(len(tok.vocab[i]) > 2 for i in ids)


def test_bpe_specials_and_unknowns():
    tok = BPETokenizer.train(TEXTS, vocab_size=200)
    assert tok.vocab[BLANK_ID] == "<blank>" and tok.vocab[SOS_ID] == "<sos/eos>"
    # Digits are stripped by normalisation, not passed through as tokens.
    assert tok.decode(tok.encode("call 42 now")) == "call now"


def test_bpe_save_load(tmp_path):
    tok = BPETokenizer.train(TEXTS, vocab_size=200)
    p = tmp_path / "bpe.json"
    tok.save(p)
    again = BPETokenizer.load(p)
    assert again.encode("the quick brown fox") == tok.encode("the quick brown fox")


# ----------------------------------------------------------------------- masks
def test_pad_mask():
    m = make_pad_mask(torch.tensor([3, 5]), 5)
    assert m[0].tolist() == [False, False, False, True, True]
    assert not m[1].any()


def test_chunk_mask_never_looks_ahead():
    m = make_chunk_mask(8, chunk_size=2, left_chunks=-1)
    # Frame 0 sees its own chunk (0,1) only.
    assert m[0].tolist() == [True, True, False, False, False, False, False, False]
    # Frame 5 sees everything up to the end of its own chunk (index 5).
    assert m[5].tolist() == [True] * 6 + [False, False]
    # Limited history: chunk 3 (frames 6,7) with 1 left chunk sees frames 4..7.
    m2 = make_chunk_mask(8, chunk_size=2, left_chunks=1)
    assert m2[6].tolist() == [False] * 4 + [True] * 4


def test_full_context_mask_is_all_true():
    assert make_chunk_mask(6, chunk_size=0).all()


# --------------------------------------------------------------------- encoder
def _encoder(**kw):
    return ConformerEncoder(n_mels=80, d_model=32, n_layers=2, n_heads=2, **kw).eval()


def test_encoder_shapes_and_lengths():
    enc = _encoder()
    feats = torch.randn(3, 200, 80)
    lens = torch.tensor([200, 150, 97])
    out, out_lens = enc(feats, lens)
    assert out.shape == (3, ((200 - 1) // 2 - 1) // 2, 32)
    assert out_lens.tolist() == [((n - 1) // 2 - 1) // 2 for n in [200, 150, 97]]


def test_padding_does_not_leak_into_real_frames():
    """A short utterance must encode identically whether or not it is batched
    with a long one. If it doesn't, a mask is wrong and every batched eval
    number is off."""
    enc = _encoder()
    short = torch.randn(1, 96, 80)
    alone, alone_len = enc(short, torch.tensor([96]))

    batched_in = torch.cat([short, torch.randn(1, 96, 80)], dim=0)
    batched_in = torch.cat([batched_in, torch.randn(2, 104, 80)], dim=1)
    out, _ = enc(batched_in, torch.tensor([96, 200]))
    n = int(alone_len[0])
    torch.testing.assert_close(out[0, :n], alone[0, :n], atol=1e-5, rtol=1e-4)


def test_chunked_encoder_is_truly_causal():
    """THE streaming contract: appending future audio must not change already
    emitted frames."""
    enc = _encoder(causal_conv=True)
    chunk = 4
    feats = torch.randn(1, 400, 80)
    full, full_len = enc(feats, torch.tensor([400]), chunk_size=chunk, left_chunks=-1)

    prefix_frames = 200
    part, part_len = enc(feats[:, :prefix_frames], torch.tensor([prefix_frames]),
                         chunk_size=chunk, left_chunks=-1)
    # Compare only whole chunks that are complete in the prefix, and drop the
    # last one: the ×4 subsampling window straddles the truncation boundary.
    n = (int(part_len[0]) // chunk - 1) * chunk
    assert n > 0
    torch.testing.assert_close(full[0, :n], part[0, :n], atol=1e-5, rtol=1e-4)


def test_non_causal_conv_breaks_causality():
    """Guards the guard: with a symmetric conv kernel the same test must FAIL,
    proving the causal test above actually tests something."""
    enc = _encoder(causal_conv=False)
    feats = torch.randn(1, 400, 80)
    full, _ = enc(feats, torch.tensor([400]), chunk_size=4)
    part, part_len = enc(feats[:, :200], torch.tensor([200]), chunk_size=4)
    n = (int(part_len[0]) // 4 - 1) * 4
    assert not torch.allclose(full[0, :n], part[0, :n], atol=1e-5)


# ----------------------------------------------------------------------- model
def _model(vocab: int = 64):
    return HybridCTCAttention(vocab_size=vocab, d_model=32, n_layers=2, n_heads=2,
                              decoder_layers=1, kernel_size=7, dropout=0.0)


def test_hybrid_loss_is_finite_and_backprops():
    m = _model()
    feats = torch.randn(2, 200, 80)
    f_lens = torch.tensor([200, 160])
    ys = torch.tensor([[5, 6, 7, 8], [9, 10, 0, 0]])
    y_lens = torch.tensor([4, 2])
    loss, parts = m(feats, f_lens, ys, y_lens)
    assert torch.isfinite(loss) and parts["ctc"] > 0 and parts["att"] > 0
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_add_sos_eos():
    ys = torch.tensor([[5, 6, 7], [9, 0, 0]])
    ys_in, ys_out = HybridCTCAttention._add_sos_eos(ys, torch.tensor([3, 1]))
    assert ys_in[0].tolist() == [SOS_ID, 5, 6, 7]
    assert ys_out[0].tolist() == [5, 6, 7, SOS_ID]
    assert ys_in[1].tolist() == [SOS_ID, 9, BLANK_ID, BLANK_ID]
    assert ys_out[1].tolist() == [9, SOS_ID, BLANK_ID, BLANK_ID]


def test_recognize_returns_token_ids():
    m = _model().eval()
    feats = torch.randn(2, 200, 80)
    out = m.recognize(feats, torch.tensor([200, 150]), beam_size=3)
    assert len(out) == 2 and all(isinstance(h, list) for h in out)


def test_ctc_beam_recovers_a_confident_path():
    """A near-deterministic emission must decode to exactly that token string."""
    t, v = 12, 6
    lp = torch.full((t, v), -20.0)
    lp[:, BLANK_ID] = -0.001
    for frame, tok in [(2, 3), (5, 4), (6, 4), (9, 5)]:   # 4 repeated => one token
        lp[frame, BLANK_ID] = -20.0
        lp[frame, tok] = -0.001
    best = ctc_prefix_beam_search(lp.log_softmax(-1), beam_size=5)[0][0]
    assert list(best) == [3, 4, 5]


def test_beam_search_survives_long_input():
    """Linear-space beam search underflows to all-zeros here; log space must not."""
    lp = torch.randn(600, 32).log_softmax(-1)
    nbest = ctc_prefix_beam_search(lp, beam_size=4)
    assert nbest and all(s > -float("inf") for _, s in nbest)


# ---------------------------------------------------------------------- metrics
def test_word_error_rate():
    assert word_error_rate("the cat sat", "the cat sat") == 0.0
    assert word_error_rate("the cat", "the cat sat") == 1 / 3
    assert corpus_wer([("a b", "a b"), ("x", "y")]) == 1 / 3


# -------------------------------------------------------------------- learning
def test_overfits_a_tiny_batch():
    """Two utterances, 120 steps, tiny model: loss must collapse. If this fails,
    the bug is in the model, not the data."""
    torch.manual_seed(0)
    m = _model(vocab=32)
    feats = torch.randn(2, 240, 80)
    f_lens = torch.tensor([240, 240])
    ys = torch.tensor([[5, 6, 7, 8, 9], [10, 11, 12, 13, 14]])
    y_lens = torch.tensor([5, 5])
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)

    first = None
    for _ in range(120):
        loss, _ = m(feats, f_lens, ys, y_lens)
        first = first if first is not None else float(loss.detach())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
    assert float(loss) < 0.35 * first, f"loss {first:.2f} -> {float(loss):.2f}"


def test_from_state_dict_rebuilds_the_exact_model():
    """The checkpoint carries no architecture config — every shape-affecting
    hyperparameter is recovered from the weights. A config file and a checkpoint
    can drift apart; tensor shapes cannot."""
    torch.manual_seed(0)
    src = HybridCTCAttention(vocab_size=48, d_model=64, n_layers=3, n_heads=8,
                             decoder_layers=2, kernel_size=15, dropout=0.0).eval()
    rebuilt = HybridCTCAttention.from_state_dict(src.state_dict())

    assert len(rebuilt.encoder.blocks) == 3
    assert len(rebuilt.decoder.layers) == 2
    assert rebuilt.encoder.blocks[0].attn.h == 8
    assert rebuilt.encoder.blocks[0].conv.kernel_size == 15
    assert rebuilt.vocab_size == 48

    feats, lens = torch.randn(1, 200, 80), torch.tensor([200])
    a, _ = src.encode(feats, lens)
    b, _ = rebuilt.encode(feats, lens)
    torch.testing.assert_close(a, b)


def test_from_state_dict_accepts_the_wrapped_checkpoint_format():
    src = HybridCTCAttention(vocab_size=48, d_model=64, n_layers=2, n_heads=4,
                             decoder_layers=1, dropout=0.0)
    wrapped = {"model": src.state_dict(), "tokenizer": "outputs/bpe_256.json",
               "causal_conv": True, "n_mels": 80}
    assert HybridCTCAttention.from_state_dict(wrapped).vocab_size == 48


def test_parameter_count_is_in_the_expected_range():
    m = HybridCTCAttention(vocab_size=512, d_model=256, n_layers=12, n_heads=4,
                           decoder_layers=4)
    n = count_parameters(m)
    assert 20e6 < n < 45e6, f"{n/1e6:.1f}M params"
