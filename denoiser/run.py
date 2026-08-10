#!/usr/bin/env python
"""CLI for the realtime voice-call denoiser.

    python run.py devices                 # list audio devices
    python run.py file  noisy.wav out.wav # denoise a file (no hardware needed)
    python run.py live  --output "BlackHole 2ch"   # live mic -> denoiser -> device

Run `python run.py live -h` for live options (device selection, attenuation limit).
"""
from __future__ import annotations

import argparse
import sys
import time


def cmd_devices(args):
    import sounddevice as sd
    print(sd.query_devices())
    try:
        din, dout = sd.default.device
        print(f"\ndefault input={din}  output={dout}")
    except Exception:
        pass
    print("\nTip: to feed a call, choose a virtual device (e.g. 'BlackHole 2ch') "
          "as --output, then select that same device as the microphone in your "
          "calling app.")


def _find_device(name_or_index, kind):
    """kind: 'input' or 'output'. Accepts an int index or a name substring."""
    import sounddevice as sd
    if name_or_index is None:
        return None
    try:
        return int(name_or_index)
    except (TypeError, ValueError):
        pass
    needle = str(name_or_index).lower()
    matches = []
    for i, d in enumerate(sd.query_devices()):
        ch = d["max_input_channels"] if kind == "input" else d["max_output_channels"]
        if ch > 0 and needle in d["name"].lower():
            matches.append((i, d["name"]))
    if not matches:
        raise SystemExit(f"no {kind} device matching {name_or_index!r}")
    if len(matches) > 1:
        opts = ", ".join(f"[{i}] {n}" for i, n in matches)
        raise SystemExit(f"ambiguous {kind} device {name_or_index!r}: {opts}")
    return matches[0][0]


def _make_backend(args):
    from denoiser.backends import make_backend
    if args.backend == "passthrough":
        return make_backend("passthrough")
    if args.backend in ("gtcrn", "dtln"):
        return make_backend(args.backend)
    return make_backend(
        "deepfilternet",
        atten_lim_db=args.atten_lim_db,
        post_filter=args.post_filter,
        hop_ms=args.hop_ms,
        context_ms=args.context_ms,
    )


def cmd_file(args):
    from denoiser.offline import denoise_file
    print(f"loading model ({args.backend})...", flush=True)
    backend = _make_backend(args)
    print(f"denoising {args.infile} -> {args.outfile}", flush=True)
    t = time.time()
    info = denoise_file(backend, args.infile, args.outfile)
    dt = time.time() - t
    dur = info["out_samples"] / info["sr"]
    print(f"done in {dt:.1f}s  ({dur:.1f}s audio, RTF={dt/max(dur,1e-9):.3f}, sr={info['sr']})")


def cmd_live(args):
    import sounddevice as sd
    from denoiser.pipeline import RealtimeDenoiser

    in_dev = _find_device(args.input, "input")
    out_dev = _find_device(args.output, "output")
    print(f"loading model ({args.backend})...", flush=True)
    backend = _make_backend(args)

    out_info = sd.query_devices(out_dev if out_dev is not None else sd.default.device[1])
    out_ch = 2 if out_info["max_output_channels"] >= 2 else 1

    in_name = sd.query_devices(in_dev if in_dev is not None else sd.default.device[0])["name"]
    print(f"input:  {in_name}")
    print(f"output: {out_info['name']} ({out_ch}ch)")

    levels = {"in": 0.0, "out": 0.0}

    def on_level(i, o):
        levels["in"], levels["out"] = i, o

    rd = RealtimeDenoiser(
        backend, in_dev, out_dev, output_channels=out_ch,
        input_gain=args.input_gain, output_gain=args.output_gain, on_level=on_level)
    rd.start()
    print(f"~added latency: {rd.latency_ms:.0f} ms   (Ctrl-C to stop)\n")

    def bar(rms, width=24):
        db = 20 * (rms if rms <= 0 else __import__("math").log10(rms))
        db = max(-60.0, db)
        fill = int((db + 60) / 60 * width)
        return "#" * fill + "-" * (width - fill)

    try:
        while True:
            time.sleep(0.1)
            sys.stdout.write(
                f"\r in [{bar(levels['in'])}]  out [{bar(levels['out'])}]  "
                f"xruns={rd.xruns}   ")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        rd.stop()


def main():
    p = argparse.ArgumentParser(description="Realtime voice-call denoiser")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_backend_opts(sp):
        sp.add_argument("--backend", choices=["deepfilternet", "gtcrn", "dtln", "passthrough"],
                        default="deepfilternet",
                        help="deepfilternet = best quality (~23%% CPU, ~50ms); "
                             "gtcrn = recommended per-frame streaming (~3%% CPU, ~32ms, phase-aware); "
                             "dtln = alternate streaming (~1.5%% CPU)")
        sp.add_argument("--atten-lim-db", type=float, default=None,
                        help="cap suppression in dB (e.g. 25 = keep some noise, "
                             "more natural). Default: full suppression.")
        sp.add_argument("--post-filter", action="store_true",
                        help="enable DeepFilterNet's extra light suppression")
        sp.add_argument("--hop-ms", type=float, default=40.0,
                        help="output granularity / added latency (default 40)")
        sp.add_argument("--context-ms", type=float, default=200.0,
                        help="warm left-context length (default 200)")

    sp = sub.add_parser("devices", help="list audio devices")
    sp.set_defaults(func=cmd_devices)

    sp = sub.add_parser("file", help="denoise a WAV file")
    sp.add_argument("infile"); sp.add_argument("outfile")
    add_backend_opts(sp)
    sp.set_defaults(func=cmd_file)

    sp = sub.add_parser("live", help="live mic -> denoiser -> output device")
    sp.add_argument("--input", default=None, help="input device name or index (default: system default)")
    sp.add_argument("--output", default=None, help="output device name or index (e.g. 'BlackHole 2ch')")
    sp.add_argument("--input-gain", type=float, default=1.0)
    sp.add_argument("--output-gain", type=float, default=1.0)
    add_backend_opts(sp)
    sp.set_defaults(func=cmd_live)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
