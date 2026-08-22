"""
Module 6, part 4: the hybrid CTC / attention recognizer.

Module 2 trained CTC alone. CTC's defining assumption is CONDITIONAL
INDEPENDENCE: given the audio, each frame's label is predicted without
reference to the labels around it. That assumption is what makes the loss a
tractable dynamic program — and it is exactly why our character model produced
"he inl san er mianthes tlen kis roint": phonetically plausible, orthographically
impossible, because nothing in the model knows that a 'q' is followed by a 'u'.

The fix is to add a decoder that IS autoregressive, and train both heads on the
same encoder (Watanabe et al. 2017; the recipe behind ESPnet and WeNet):

    feats ─▶ Conformer encoder ─┬─▶ CTC head        ─▶ L_ctc     (alignment)
                                └─▶ Transformer dec ─▶ L_att     (language)

    L = λ·L_ctc + (1-λ)·L_att,   λ ≈ 0.3

Each head fixes the other's failure mode. CTC is monotonic by construction, so
it stops attention from wandering or looping — the exact failure Module 4's TTS
needed guided attention to avoid. The decoder supplies the implicit language
model CTC lacks. And you get both for one encoder's worth of compute.

DECODING gets the same benefit, in two passes:
  1. CTC prefix beam search produces an n-best list. Fast, streaming-compatible,
     and it never proposes a non-monotonic hypothesis.
  2. The decoder RESCORES those n hypotheses in a single batched forward.
This is WeNet's "attention rescoring", and it is much cheaper than running an
autoregressive beam search — the decoder is evaluated once per hypothesis, not
once per output token per beam.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.asr.conformer import ConformerEncoder, make_pad_mask
from src.asr.search import ctc_greedy, ctc_prefix_beam_search
from src.asr.tokenizer import BLANK_ID, SOS_ID


class MultiHeadAttention(nn.Module):
    """Plain scaled dot-product attention (the decoder needs no relative
    positions — its sequence is short and its positions are literal)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.h, self.dk = n_heads, d_model // n_heads
        self.q, self.k, self.v = (nn.Linear(d_model, d_model) for _ in range(3))
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        b, lq, _ = query.shape
        lk = key.size(1)
        q = self.q(query).view(b, lq, self.h, self.dk).transpose(1, 2)
        k = self.k(key).view(b, lk, self.h, self.dk).transpose(1, 2)
        v = self.v(value).view(b, lk, self.h, self.dk).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(b, lq, self.h * self.dk)
        return self.out(ctx)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.pe[:, : x.size(1)].to(x.dtype)


class DecoderLayer(nn.Module):
    """Pre-norm: self-attention over emitted tokens, then cross-attention over
    the encoder, then a feed-forward. Pre-norm because post-norm transformers
    need a warmup schedule to not diverge, and we have little data to waste."""

    def __init__(self, d_model: int, n_heads: int, ff: int, dropout: float):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, self_mask, memory_mask):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, self_mask))
        h = self.norm2(x)
        x = x + self.dropout(self.cross_attn(h, memory, memory, memory_mask))
        return x + self.dropout(self.ff(self.norm3(x)))


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 4,
                 n_heads: int = 4, ff_expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=BLANK_ID)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_model * ff_expansion, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, ys_in: torch.Tensor, memory: torch.Tensor,
                memory_lengths: torch.Tensor) -> torch.Tensor:
        b, l = ys_in.shape
        x = self.pos(self.embed(ys_in))
        # Causal: token i may attend to 0..i. Without this the decoder simply
        # reads the answer and the loss goes to zero while WER stays at 100%.
        causal = torch.tril(torch.ones(l, l, dtype=torch.bool, device=ys_in.device))
        self_mask = causal.unsqueeze(0).unsqueeze(0)
        mem_mask = (~make_pad_mask(memory_lengths, memory.size(1)))
        mem_mask = mem_mask.unsqueeze(1).unsqueeze(1)          # (B,1,1,T)
        for layer in self.layers:
            x = layer(x, memory, self_mask, mem_mask)
        return self.out(self.norm(x))


# ---------------------------------------------------------------------------
class HybridCTCAttention(nn.Module):
    def __init__(self, vocab_size: int, n_mels: int = 80, d_model: int = 256,
                 n_layers: int = 12, n_heads: int = 4, decoder_layers: int = 4,
                 kernel_size: int = 31, causal_conv: bool = True,
                 ctc_weight: float = 0.3, label_smoothing: float = 0.1,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = ConformerEncoder(
            n_mels=n_mels, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            kernel_size=kernel_size, causal_conv=causal_conv, dropout=dropout,
        )
        self.ctc_head = nn.Linear(d_model, vocab_size)
        self.decoder = TransformerDecoder(
            vocab_size, d_model, decoder_layers, n_heads, dropout=dropout
        )
        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight
        self.ctc_loss = nn.CTCLoss(blank=BLANK_ID, reduction="mean", zero_infinity=True)
        self.att_loss = nn.CrossEntropyLoss(
            ignore_index=BLANK_ID, label_smoothing=label_smoothing
        )

    # ---- loading --------------------------------------------------------
    @classmethod
    def from_state_dict(cls, state: dict, causal_conv: bool = True
                        ) -> "HybridCTCAttention":
        """Rebuild the model from weights alone — every hyperparameter that
        affects a tensor SHAPE is recoverable from the shapes themselves.

        Worth doing rather than storing a config blob: a config file and a
        checkpoint can drift apart, shapes cannot. (`causal_conv` is the one
        exception — it changes padding, not shapes — so it stays an argument.)
        """
        sd = state.get("model", state)
        n_blocks = 1 + max(
            int(k.split(".")[2]) for k in sd if k.startswith("encoder.blocks.")
        )
        n_dec = 1 + max(
            int(k.split(".")[2]) for k in sd if k.startswith("decoder.layers.")
        )
        vocab, d_model = sd["ctc_head.weight"].shape
        n_heads = sd["encoder.blocks.0.attn.u_bias"].shape[0]
        kernel = sd["encoder.blocks.0.conv.depthwise.weight"].shape[-1]
        ff_expansion = sd["encoder.blocks.0.ff1.net.1.weight"].shape[0] // d_model
        n_mels = state.get("n_mels", 80)

        model = cls(
            vocab_size=vocab, n_mels=n_mels, d_model=d_model, n_layers=n_blocks,
            n_heads=n_heads, decoder_layers=n_dec, kernel_size=kernel,
            causal_conv=state.get("causal_conv", causal_conv), dropout=0.0,
        )
        assert ff_expansion == 4, f"unexpected ff expansion {ff_expansion}"
        model.load_state_dict(sd)
        return model.eval()

    # ---- training -------------------------------------------------------
    @staticmethod
    def _add_sos_eos(ys: torch.Tensor, ys_lens: torch.Tensor):
        """y -> (ys_in = [sos] y, ys_out = y [eos]), padded with BLANK_ID.

        One token serves as both sos and eos (ESPnet convention): the decoder
        is started from it and trained to emit it, so 'when to stop' is learned
        rather than heuristic — the thing Module 4's TTS stop head had to guard
        with hand-written rules.
        """
        b, l = ys.shape
        ys_in = ys.new_full((b, l + 1), BLANK_ID)
        ys_out = ys.new_full((b, l + 1), BLANK_ID)
        ys_in[:, 0] = SOS_ID
        for i in range(b):
            n = int(ys_lens[i])
            ys_in[i, 1:n + 1] = ys[i, :n]
            ys_out[i, :n] = ys[i, :n]
            ys_out[i, n] = SOS_ID
        return ys_in, ys_out

    def _ctc(self, log_probs, ys, enc_lens, ys_lens):
        """CTC loss, with an Apple-Silicon detour.

        `aten::_ctc_loss` has no MPS kernel (PyTorch #141287), so on Apple GPUs
        we run just this one op on the CPU. Autograd stitches the devices
        together transparently. We do this explicitly rather than setting
        PYTORCH_ENABLE_MPS_FALLBACK=1, because that flag silently relocates
        *any* missing op — turning a 10% slowdown you understand into a 10x one
        you don't. The transfer is a few MB per step and is not the bottleneck.
        """
        lp = log_probs.transpose(0, 1)                 # CTCLoss wants (T,B,V)
        if lp.device.type == "mps":
            return self.ctc_loss(lp.cpu(), ys.cpu(), enc_lens.cpu(), ys_lens.cpu())
        return self.ctc_loss(lp, ys, enc_lens, ys_lens)

    def forward(self, feats, feat_lens, ys, ys_lens, chunk_size: int = 0,
                left_chunks: int = -1):
        enc, enc_lens = self.encoder(feats, feat_lens, chunk_size, left_chunks)

        log_probs = F.log_softmax(self.ctc_head(enc), dim=-1)
        loss_ctc = self._ctc(log_probs, ys, enc_lens, ys_lens).to(enc.device)

        ys_in, ys_out = self._add_sos_eos(ys, ys_lens)
        logits = self.decoder(ys_in, enc, enc_lens)
        loss_att = self.att_loss(logits.reshape(-1, self.vocab_size), ys_out.reshape(-1))

        loss = self.ctc_weight * loss_ctc + (1.0 - self.ctc_weight) * loss_att
        return loss, {"ctc": float(loss_ctc.detach()), "att": float(loss_att.detach())}

    # ---- inference ------------------------------------------------------
    @torch.no_grad()
    def encode(self, feats, feat_lens, chunk_size: int = 0, left_chunks: int = -1):
        return self.encoder(feats, feat_lens, chunk_size, left_chunks)

    @torch.no_grad()
    def recognize(self, feats, feat_lens, beam_size: int = 10,
                  ctc_weight: float = 0.5, chunk_size: int = 0,
                  left_chunks: int = -1, rescore: bool = True) -> list[list[int]]:
        """Batch of utterances -> token ids. CTC beam, then attention rescoring."""
        enc, enc_lens = self.encode(feats, feat_lens, chunk_size, left_chunks)
        log_probs = F.log_softmax(self.ctc_head(enc), dim=-1)

        out: list[list[int]] = []
        for i in range(enc.size(0)):
            n = int(enc_lens[i])
            lp = log_probs[i, :n]
            if not rescore:
                out.append(ctc_greedy(lp, BLANK_ID))
                continue
            nbest = ctc_prefix_beam_search(lp, beam_size, BLANK_ID)
            nbest = [(h, s) for h, s in nbest if h] or [((), 0.0)]
            out.append(self._rescore(nbest, enc[i:i + 1, :n], ctc_weight))
        return out

    def _rescore(self, nbest, memory, ctc_weight: float) -> list[int]:
        """Score all hypotheses in ONE batched decoder forward."""
        device = memory.device
        n = len(nbest)
        max_len = max(len(h) for h, _ in nbest)
        ys = torch.full((n, max_len), BLANK_ID, dtype=torch.long, device=device)
        lens = torch.zeros(n, dtype=torch.long, device=device)
        for i, (h, _) in enumerate(nbest):
            ys[i, : len(h)] = torch.tensor(h, dtype=torch.long, device=device)
            lens[i] = len(h)

        ys_in, ys_out = self._add_sos_eos(ys, lens)
        mem = memory.repeat(n, 1, 1)
        mem_lens = torch.full((n,), memory.size(1), dtype=torch.long, device=device)
        logp = F.log_softmax(self.decoder(ys_in, mem, mem_lens), dim=-1)
        chosen = logp.gather(2, ys_out.unsqueeze(2)).squeeze(2)   # (N, L)
        valid = torch.arange(ys_out.size(1), device=device).unsqueeze(0) <= lens.unsqueeze(1)
        att_scores = (chosen * valid).sum(dim=1)                  # include the eos step

        best, best_score = nbest[0][0], -float("inf")
        for i, (h, ctc_score) in enumerate(nbest):
            score = ctc_weight * ctc_score + (1 - ctc_weight) * float(att_scores[i])
            if score > best_score:
                best, best_score = h, score
        return list(best)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
