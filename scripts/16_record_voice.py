"""
Module 8, part 4: record your voice, one prompt at a time.

    uv run python scripts/16_record_voice.py

This is the only step in the whole book that cannot be parallelised, bought, or
downloaded, so it is the one worth building a good tool for. An hour of speech
is ~900 takes; at ten seconds of friction each that is two and a half hours of
overhead, which is how voice datasets end up unfinished.

The design is therefore about removing decisions:

  HANDS FREE. Press ENTER, read the sentence, stop talking. The Module 1 VAD
  hears you stop and ends the take — no second keypress, no mouse. Your hands
  never leave the return key for 900 sentences.

  EVERY TAKE IS CHECKED BEFORE IT COUNTS (`src/tts/qc.py`). Clipping, noise
  floor, silence padding, and a transcription of what you actually said versus
  what was on screen. Bad takes are caught while re-recording costs five
  seconds, not after training when it costs a day.

  RESUMABLE AT ANY INSTANT. Accepted takes are flushed immediately and the
  manifest is append-only, keyed on text. Close the laptop mid-sentence and you
  lose that sentence. Nothing else.

WHAT THIS SCRIPT IS QUIETLY REUSING. The VAD that decides when you stopped
talking is Module 1, trained in `scripts/09_train_vad_real.py`. The features it
runs on are Module 0. The recording verifier is Module 6's Conformer if you ask
for it (`--asr scratch`), and Whisper otherwise — see `src/tts/qc.py` for why
that default is not laziness. The corpus this writes is what Module 8's
acoustic model and Module 7's vocoder will be fine-tuned on.

RECORDING ADVICE THAT MATTERS MORE THAN ANY HYPERPARAMETER:

  One room, one microphone, one distance, one time of day. The model cannot
  separate "your voice" from "your recording chain", so any change to the chain
  becomes a change to the voice. Consistency beats quality: a mediocre mic used
  identically for 900 takes trains a better voice than a great mic moved
  halfway through.

  A hand's width from the mic, slightly off-axis so plosives miss the capsule.

  Speak the way you want to be replied to. If you read this like an audiobook,
  your agent will sound like an audiobook. The conversational prompts are there
  to be said, not recited.

  Stop at the first sign of vocal fatigue. Twenty tired minutes will do more
  damage than the twenty minutes are worth, because the model averages
  everything it is given.
"""

from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.features import log_mel_spectrogram  # noqa: E402
from src.tts.corpus import MODEL_SR, VoiceCorpus, resample  # noqa: E402
from src.tts.prompts import load_or_build_prompts  # noqa: E402
from src.tts.qc import check_take  # noqa: E402
from src.vad.model import VADNet  # noqa: E402
from src.vad.stream import HysteresisGate  # noqa: E402

BLOCK_MS = 100.0          # capture granularity; also the VAD decision cadence
FRAME, HOP = 400, 160     # log_mel_spectrogram's 25 ms / 10 ms grid at 16 kHz
TRAIL_SILENCE_S = 0.8     # how long you must be quiet before the take ends
LEAD_TIMEOUT_S = 6.0      # give up waiting for speech to start
MAX_TAKE_S = 20.0         # hard cap so a stuck VAD cannot record forever

DIM, BOLD, GREEN, YELLOW, RED, RESET = (
    "\033[2m", "\033[1m", "\033[32m", "\033[33m", "\033[31m", "\033[0m")


def load_vad(path: str = "outputs/vad_real.pt") -> VADNet:
    p = Path(path)
    if not p.exists():
        p = Path("outputs/vad.pt")
    if not p.exists():
        sys.exit("no VAD checkpoint — run: uv run python scripts/09_train_vad_real.py")
    model = VADNet(n_mels=80)
    model.load_state_dict(torch.load(p, map_location="cpu"))
    print(f"{DIM}[vad] {p}{RESET}")
    return model.eval()


def make_transcriber(kind: str):
    """The read-check. See src/tts/qc.py on why whisper is the default."""
    if kind == "none":
        return None
    if kind == "whisper":
        from faster_whisper import WhisperModel
        model = WhisperModel("base.en", device="cpu", compute_type="int8")
        print(f"{DIM}[qc] verifier: whisper base.en int8{RESET}")

        def transcribe(audio: np.ndarray) -> str:
            segments, _ = model.transcribe(audio, beam_size=1, language="en")
            return " ".join(s.text for s in segments)
        return transcribe

    from src.asr.hybrid import HybridCTCAttention
    from src.asr.tokenizer import BPETokenizer
    state = torch.load("outputs/asr_conformer.pt", map_location="cpu")
    model = HybridCTCAttention.from_state_dict(state)
    tok = BPETokenizer.load(state["tokenizer"])
    print(f"{YELLOW}[qc] verifier: your Module 6 Conformer — expect false "
          f"alarms on an unseen speaker (see src/tts/qc.py){RESET}")

    def transcribe(audio: np.ndarray) -> str:
        mel = log_mel_spectrogram(torch.from_numpy(audio), sr=MODEL_SR)
        ids = model.recognize(mel.unsqueeze(0), torch.tensor([mel.shape[0]]))
        return tok.decode(ids[0])
    return transcribe


class Recorder:
    """Capture until the VAD says you stopped talking.

    Runs the VAD on the LIVE stream rather than post-hoc so the take ends the
    moment you finish, which is the whole ergonomic difference between this and
    a two-keypress recorder. The audio kept is the full-rate original; the VAD
    gets a 16 kHz copy because that is what it was trained on.
    """

    def __init__(self, vad: VADNet, device_sr: int = 48_000):
        import sounddevice as sd

        self.sd = sd
        self.sr = device_sr
        self.vad = vad
        self.block = int(device_sr * BLOCK_MS / 1000)

    def record(self) -> tuple[np.ndarray, str]:
        q: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, frames, time_info, status):
            q.put(indata[:, 0].copy())

        chunks: list[np.ndarray] = []
        gate = HysteresisGate()
        pending = np.zeros(0, dtype=np.float32)   # 16 kHz tail awaiting a full frame
        heard_speech = False
        silence_s = 0.0
        t0 = time.perf_counter()

        with self.sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                 blocksize=self.block, callback=callback):
            while True:
                try:
                    block = q.get(timeout=1.0)
                except queue.Empty:
                    return np.zeros(0, dtype=np.float32), "no audio from the device"
                chunks.append(block)

                # VAD path: 16 kHz. Framing has to be CONTINUOUS across
                # blocks, which means carrying a (frame - hop) = 240-sample
                # overlap tail, not just the leftover hop. Frame each block
                # independently and you silently drop the 15 ms straddling
                # every boundary — 15% of all frames, biased toward exactly
                # the onsets and offsets the VAD is looking for.
                pending = np.concatenate([pending, resample(block, self.sr, MODEL_SR)])
                if len(pending) >= FRAME:
                    n_frames = (len(pending) - FRAME) // HOP + 1
                    usable = pending[:FRAME + (n_frames - 1) * HOP]
                    pending = pending[n_frames * HOP:]
                    mel = log_mel_spectrogram(torch.from_numpy(usable), sr=MODEL_SR)
                    if mel.shape[0]:
                        with torch.no_grad():
                            probs = torch.sigmoid(self.vad(mel.unsqueeze(0))[0])
                        for p in probs.tolist():
                            if gate.update(float(p)):
                                heard_speech, silence_s = True, 0.0
                            elif heard_speech:
                                silence_s += 0.01

                elapsed = time.perf_counter() - t0
                peak = float(np.abs(block).max())
                bar = "█" * int(min(peak, 1.0) * 28)
                state = "listening" if not heard_speech else "recording"
                print(f"\r  {state:9s} {elapsed:5.1f}s |{bar:<28s}| ", end="", flush=True)

                if heard_speech and silence_s >= TRAIL_SILENCE_S:
                    break
                if not heard_speech and elapsed > LEAD_TIMEOUT_S:
                    print("\r" + " " * 60, end="\r")
                    return np.concatenate(chunks), "heard nothing"
                if elapsed > MAX_TAKE_S:
                    break

        print("\r" + " " * 60, end="\r")
        return np.concatenate(chunks), ""


def report(qc) -> None:
    flag = f"{GREEN}ok{RESET}" if qc.ok else f"{RED}reject{RESET}"
    print(f"  {flag}  {qc.duration_s:.1f}s  peak {qc.peak:.2f}  "
          f"snr {qc.snr_db:.0f} dB" + (f"  cer {qc.cer:.2f}" if qc.heard else ""))
    for p in qc.problems:
        print(f"    {RED}x{RESET} {p}")
    for w in qc.warnings:
        print(f"    {YELLOW}!{RESET} {w}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Record a single-speaker TTS corpus.")
    ap.add_argument("--minutes", type=float, default=60.0,
                    help="target corpus length; only used when building prompts")
    ap.add_argument("--asr", choices=["whisper", "scratch", "none"], default="whisper",
                    help="read-check engine (see src/tts/qc.py)")
    ap.add_argument("--device-sr", type=int, default=48_000,
                    help="capture rate; stored at 24 kHz regardless")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    prompts = load_or_build_prompts(target_minutes=args.minutes)
    corpus = VoiceCorpus()
    done = corpus.recorded_texts()
    todo = [p for p in prompts if p not in done]

    recorded_s = corpus.total_seconds()
    print(f"\n{BOLD}Module 8 — recording your voice{RESET}")
    print(f"  corpus     {len(done)}/{len(prompts)} takes, "
          f"{recorded_s / 60:.1f} min on disk")
    if not todo:
        print(f"\n{GREEN}Corpus complete.{RESET}\n")
        return

    vad = load_vad()
    transcribe = make_transcriber(args.asr)
    rec = Recorder(vad, args.device_sr)

    print(f"\n  {DIM}ENTER record · r retake · s skip · q quit{RESET}")
    print(f"  {DIM}stop talking and the take ends by itself{RESET}\n")

    session_start = time.perf_counter()
    session_takes = 0
    i = 0
    while i < len(todo):
        prompt = todo[i]
        n_done = len(prompts) - len(todo) + i
        print(f"{DIM}{n_done + 1}/{len(prompts)}  "
              f"{recorded_s / 60:.1f} min{RESET}")
        print(f"  {BOLD}{prompt}{RESET}")

        key = input("  > ").strip().lower()
        if key == "q":
            break
        if key == "s":
            i += 1
            print()
            continue

        audio, err = rec.record()
        if err:
            print(f"  {YELLOW}{err} — retrying{RESET}\n")
            continue

        trimmed, qc = check_take(audio, rec.sr, prompt, vad, transcribe)
        report(qc)
        if trimmed is None or not qc.ok:
            print(f"  {DIM}retaking{RESET}\n")
            continue

        # Warnings are yours to overrule: only you can tell a genuine misread
        # from the verifier mishearing a name.
        if qc.warnings:
            if input("  keep it? [Y/r] ").strip().lower() == "r":
                print()
                continue

        corpus.append(trimmed, rec.sr, prompt, prompt)
        recorded_s += qc.duration_s
        session_takes += 1
        i += 1
        print()

    mins = recorded_s / 60
    elapsed = (time.perf_counter() - session_start) / 60
    print(f"\n{BOLD}Session{RESET}")
    print(f"  {session_takes} takes in {elapsed:.0f} min")
    print(f"  corpus now {mins:.1f} min over {len(corpus.load())} utterances")
    if session_takes:
        rate = elapsed / session_takes
        left = len(todo) - session_takes
        print(f"  {left} prompts left  (~{left * rate / 60:.1f} h at this pace)")
    print(f"  {DIM}data/voice/{RESET}\n")


if __name__ == "__main__":
    main()
