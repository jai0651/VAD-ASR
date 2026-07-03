"""
Module 4, part 2: the acoustic model — text to log-mel, one frame at a time.

This is a mini Tacotron. ASR's alignment problem, mirrored: there we had many
audio frames per character and CTC discovered the alignment; here we must
DECIDE how long each character sounds while generating. Tacotron's answer is
an autoregressive decoder with ATTENTION over the text:

    every decoder step:                                (one step = r mel frames)
      1. prenet(previous mel frame)             "what did I just say?"
      2. attention over encoder states          "which characters am I on?"
      3. GRU state update                       "remember where I am"
      4. project [state, context] -> r frames   "paint the next slice of mel"
      5. stop head -> P(utterance finished)     "am I done talking?"

The three design choices that make it trainable at this scale:

  LOCATION-SENSITIVE ATTENTION. Content-only attention can stutter (re-attend
  the same syllable) or skip. Feeding the *previous* alignment through a conv
  gives the scorer a sense of "I was at character 12, so look near 13" —
  a soft monotonicity prior. Speech is read left to right; the model should
  know that.

  REDUCTION FACTOR r=2. Predict 2 frames per decoder step: halves the loop
  length and, more subtly, forces each attention step to cover more audio,
  which makes alignments form much faster.

  ALWAYS-ON PRENET DROPOUT. The decoder sees its previous frame; with a
  perfect memory it would just copy-forward and ignore the text. Dropout on
  that path (kept on even at inference, as in Tacotron 2) keeps the decoder
  dependent on attention — remove it and you get mumble collapse.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.asr.text import VOCAB_SIZE

N_MELS = 80


class Prenet(nn.Module):
    def __init__(self, in_dim: int = N_MELS, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # training=True unconditionally: see module docstring.
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=True)
        return F.dropout(F.relu(self.fc2(x)), p=0.5, training=True)


class LocationAttention(nn.Module):
    """score_i = v · tanh(W_q·state + W_k·enc_i + W_f·conv(prev_alignment)_i)"""

    def __init__(self, enc_dim: int = 256, dec_dim: int = 256, attn_dim: int = 128):
        super().__init__()
        self.query = nn.Linear(dec_dim, attn_dim, bias=False)
        self.keys = nn.Linear(enc_dim, attn_dim, bias=False)
        self.loc_conv = nn.Conv1d(1, 32, kernel_size=31, padding=15, bias=False)
        self.loc_proj = nn.Linear(32, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, state, enc_keys, enc_out, prev_align, text_mask):
        # enc_keys = self.keys(enc_out), precomputed once per utterance.
        loc = self.loc_conv(prev_align.unsqueeze(1)).transpose(1, 2)  # (B, T_in, 32)
        scores = self.v(
            torch.tanh(self.query(state).unsqueeze(1) + enc_keys + self.loc_proj(loc))
        ).squeeze(-1)                                                 # (B, T_in)
        scores = scores.masked_fill(~text_mask, float("-inf"))       # ignore padding
        align = torch.softmax(scores, dim=-1)
        context = torch.bmm(align.unsqueeze(1), enc_out).squeeze(1)   # (B, enc_dim)
        return context, align


class TacotronMini(nn.Module):
    def __init__(self, vocab: int = VOCAB_SIZE, r: int = 2):
        super().__init__()
        self.r = r
        self.embedding = nn.Embedding(vocab, 128, padding_idx=0)
        self.encoder = nn.GRU(128, 128, batch_first=True, bidirectional=True)
        self.prenet = Prenet()
        self.attention = LocationAttention()
        self.decoder_cell = nn.GRUCell(128 + 256, 256)
        self.mel_proj = nn.Linear(256 + 256, N_MELS * r)
        self.stop_proj = nn.Linear(256 + 256, r)

    def encode(self, text: torch.Tensor):
        enc_out, _ = self.encoder(self.embedding(text))   # (B, T_in, 256)
        return enc_out, self.attention.keys(enc_out)

    def _step(self, prev_frame, state, enc_keys, enc_out, prev_align, text_mask):
        """One decoder step: previous frame + attention -> r new frames."""
        pre = self.prenet(prev_frame)                                  # (B, 128)
        context, align = self.attention(
            state, enc_keys, enc_out, prev_align, text_mask
        )
        state = self.decoder_cell(torch.cat([pre, context], dim=-1), state)
        both = torch.cat([state, context], dim=-1)
        frames = self.mel_proj(both).view(-1, self.r, N_MELS)          # (B, r, 80)
        stop = self.stop_proj(both)                                    # (B, r)
        return frames, stop, state, align

    def forward(self, text, text_mask, mel_targets):
        """Teacher-forced training pass.

        text: (B, T_in) label ids; text_mask: (B, T_in) True where real;
        mel_targets: (B, T_mel, 80) with T_mel a multiple of r.
        Returns mel_out (B, T_mel, 80), stop_logits (B, T_mel), alignments.
        """
        b, t_mel, _ = mel_targets.shape
        enc_out, enc_keys = self.encode(text)

        state = torch.zeros(b, 256, device=text.device)
        align = torch.zeros(b, text.shape[1], device=text.device)
        align[:, 0] = 1.0            # start attending at the first character
        prev = torch.zeros(b, N_MELS, device=text.device)  # <go> frame

        mels, stops, aligns = [], [], []
        for t in range(t_mel // self.r):
            frames, stop, state, align = self._step(
                prev, state, enc_keys, enc_out, align, text_mask
            )
            mels.append(frames)
            stops.append(stop)
            aligns.append(align)
            # Teacher forcing: next step sees the TRUE last frame of this
            # group, so early training isn't derailed by its own bad output.
            prev = mel_targets[:, (t + 1) * self.r - 1]

        return (
            torch.cat(mels, dim=1),                       # (B, T_mel, 80)
            torch.cat(stops, dim=1),                      # (B, T_mel)
            torch.stack(aligns, dim=1),                   # (B, steps, T_in)
        )

    @torch.no_grad()
    def generate(self, text: torch.Tensor, max_frames: int = 1000):
        """Free-running inference for ONE utterance (B=1): feed our own output
        back in until the stop head fires.

        Teacher forcing has a price, and this method pays it: at inference the
        decoder eats its own imperfect frames (exposure bias), and attention
        can stall on one character or wander. Three decode-time guards — all
        classic deployed-Tacotron tricks, none require retraining:

          STOP GUARDS   a minimum length of ~5 frames/char (nobody says 80
                        characters in 2 s) and two consecutive stop votes
                        before we believe the stop head.

        (We tried harder decode-time hacks — hard attention windowing, forced
        stall-advance — and measured them WORSE with the ASR judge: the model
        was trained on full-softmax attention, and renormalizing inside a
        window hands it context vectors it has never seen. The honest fixes
        for exposure bias live in training, not decoding: more data, scheduled
        sampling, or non-autoregressive architectures. See docs/04-tts.html.)
        """
        t_in = text.shape[1]
        text_mask = torch.ones(1, t_in, dtype=torch.bool, device=text.device)
        enc_out, enc_keys = self.encode(text)
        state = torch.zeros(1, 256, device=text.device)
        align = torch.zeros(1, t_in, device=text.device)
        align[:, 0] = 1.0
        prev = torch.zeros(1, N_MELS, device=text.device)

        min_frames = 5 * t_in
        stop_votes = 0
        mels, aligns = [], []
        for step in range(max_frames // self.r):
            frames, stop, state, align = self._step(
                prev, state, enc_keys, enc_out, align, text_mask
            )
            mels.append(frames)
            aligns.append(align)
            prev = frames[:, -1]
            stop_votes = stop_votes + 1 if torch.sigmoid(stop).max() > 0.5 else 0
            if stop_votes >= 2 and (step + 1) * self.r >= min_frames:
                break
        return torch.cat(mels, dim=1)[0], torch.stack(aligns, dim=1)[0]


def guided_attention_penalty(
    aligns: torch.Tensor, text_mask: torch.Tensor, mel_mask: torch.Tensor, r: int,
    g: float = 0.2,
) -> torch.Tensor:
    """Soft diagonal prior on attention (DC-TTS): penalize attention mass far
    from the diagonal of the (decoder step, text position) map.

        W[t, n] = 1 - exp(-(n/N - t/T)^2 / (2 g^2)),   penalty = mean(A * W)

    Near the diagonal W ≈ 0 (free); far away W ≈ 1 (costly). This encodes the
    one thing we know for sure about read speech — time moves through the text
    monotonically — and cuts alignment-convergence time dramatically on small
    data. It's a *prior*, not a constraint: with enough data you can drop it.

    aligns: (B, steps, T_in); masks give each item's true lengths.
    """
    b, steps, t_in = aligns.shape
    n_text = text_mask.sum(dim=1).clamp(min=1)                 # (B,)
    n_steps = (mel_mask.sum(dim=1) / r).clamp(min=1)           # (B,)
    t_pos = torch.arange(steps, device=aligns.device).view(1, steps, 1)
    n_pos = torch.arange(t_in, device=aligns.device).view(1, 1, t_in)
    rel_t = t_pos / n_steps.view(b, 1, 1)
    rel_n = n_pos / n_text.view(b, 1, 1)
    w = 1.0 - torch.exp(-((rel_n - rel_t) ** 2) / (2 * g * g))
    # Only count real (unpadded) decoder steps and text positions.
    valid = (t_pos < n_steps.view(b, 1, 1)) & text_mask.unsqueeze(1)
    return (aligns * w * valid).sum() / valid.sum().clamp(min=1)
