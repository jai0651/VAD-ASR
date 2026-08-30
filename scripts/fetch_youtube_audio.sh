#!/usr/bin/env bash
# Module 9: pull audio off YouTube (or any yt-dlp-supported URL) into the exact
# format the rest of this repo expects — 16 kHz, mono, 16-bit PCM wav.
#
#   bash scripts/fetch_youtube_audio.sh <url> [name]
#   bash scripts/fetch_youtube_audio.sh 'https://youtu.be/XXXX' interview
#
# Writes data/tapes/<name>.wav. A podcast or interview is the ideal test tape:
# several speakers, real turn-taking, real overlap, real room acoustics — all
# the things LibriSpeech does not have.
#
# Needs yt-dlp, which is deliberately NOT a project dependency (it needs
# updating far more often than anything in pyproject.toml):
#
#   uv tool install yt-dlp        # or: brew install yt-dlp
#
# For your own recordings just convert them directly and skip this script:
#   ffmpeg -i mymeeting.m4a -ac 1 -ar 16000 -c:a pcm_s16le data/tapes/mymeeting.wav
set -euo pipefail
cd "$(dirname "$0")/.."

url="${1:-}"
name="${2:-tape}"
if [[ -z "$url" ]]; then
  echo "usage: bash scripts/fetch_youtube_audio.sh <url> [name]" >&2
  exit 2
fi
for tool in yt-dlp ffmpeg; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: $tool not found." >&2
    [[ "$tool" == "yt-dlp" ]] && echo "  install it with: uv tool install yt-dlp" >&2
    [[ "$tool" == "ffmpeg" ]] && echo "  install it with: brew install ffmpeg" >&2
    exit 1
  }
done

mkdir -p data/tapes
out="data/tapes/${name}.wav"

# -x extract audio; the postprocessor args force the sample rate / channel
# count / codec, so nothing downstream has to resample. 16 kHz mono matches
# src/audio/features.py and every model in this repo.
yt-dlp -x --audio-format wav \
  --postprocessor-args "-ac 1 -ar 16000 -c:a pcm_s16le" \
  -o "data/tapes/${name}.%(ext)s" \
  "$url"

if [[ ! -f "$out" ]]; then
  echo "error: expected $out but it was not produced" >&2
  exit 1
fi

python3 - "$out" <<'PY'
import sys, wave
with wave.open(sys.argv[1]) as w:
    ch, rate, n = w.getnchannels(), w.getframerate(), w.getnframes()
print(f"{sys.argv[1]}: {n/rate/60:.1f} min, {rate} Hz, {ch} channel(s)")
assert rate == 16000 and ch == 1, "expected 16 kHz mono — check the ffmpeg args"
PY
echo "ready. this is a mixed tape: use it with the diarizer (Module 9, step 3)."
