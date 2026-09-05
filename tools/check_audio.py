"""Check whether a recording was damaged by the browser's audio processing.

Chrome's noise suppression and echo cancellation gate far-field audio — a phone
on speakerphone — to *digital silence*. A real microphone in a real room never
produces exactly-zero samples, so a high zero-ratio is proof the client's side
was filtered out rather than merely recorded quietly.

    .venv/Scripts/python.exe tools/check_audio.py            # every interview
    .venv/Scripts/python.exe tools/check_audio.py data/2026-...-acme
"""

from __future__ import annotations

import sys
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000

# A clean far-field recording sits near 0%. Anything above this is gating.
ZERO_RATIO_LIMIT = 2.0


def load_mono(path: Path) -> np.ndarray:
    container = av.open(str(path))
    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=SAMPLE_RATE
    )
    chunks = [
        out.to_ndarray().reshape(-1)
        for frame in container.decode(audio=0)
        for out in resampler.resample(frame)
    ]
    container.close()
    if not chunks:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(chunks)


def longest_zero_run(is_zero: np.ndarray) -> int:
    if not is_zero.any():
        return 0
    # Run lengths via the gaps between non-zero positions.
    idx = np.flatnonzero(~is_zero)
    padded = np.concatenate(([-1], idx, [len(is_zero)]))
    return int((np.diff(padded) - 1).max())


def report(directory: Path) -> bool:
    audio = directory / "audio.webm"
    if not audio.is_file():
        return True

    samples = load_mono(audio)
    if samples.size == 0:
        print(f"{directory.name}: no decodable audio")
        return False

    x = samples.astype(np.float32) / 32768.0
    is_zero = samples == 0
    zero_ratio = float(is_zero.mean() * 100)
    run = longest_zero_run(is_zero) / SAMPLE_RATE

    window = int(0.05 * SAMPLE_RATE)
    frames = x[: x.size // window * window].reshape(-1, window)
    db = 20 * np.log10(np.maximum(np.sqrt((frames ** 2).mean(axis=1)), 1e-7))
    floor = float(np.percentile(db, 10))

    # Level of the material that is actually above the floor, ignoring transients.
    active = db[db > np.percentile(db, 60)]
    body = float(active.mean()) if active.size else -99.0

    # Speech lives in 300-3400 Hz. Hiss and rumble do not. A recording whose
    # energy sits outside that band contains room noise, not a conversation.
    n = 2048
    usable = x[x != 0]
    speech_share = 0.0
    if usable.size >= n:
        seg = usable[: usable.size // n * n].reshape(-1, n)
        spec = np.abs(np.fft.rfft(seg * np.hanning(n), axis=1)).mean(axis=0)
        freq = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
        band = (freq >= 300) & (freq < 3400)
        speech_share = float(spec[band].sum() / max(spec.sum(), 1e-12) * 100)

    if speech_share < 45:
        verdict = "NO SPEECH — mostly hiss/rumble; the voice never reached the mic"
        ok = False
    elif body < -45:
        verdict = "TOO FAINT — speech present but below what Whisper can resolve"
        ok = False
    elif zero_ratio > ZERO_RATIO_LIMIT:
        verdict = "GATED — audio was filtered out by the browser or the driver"
        ok = False
    else:
        verdict = "OK — usable recording"
        ok = True

    print(f"\n{directory.name}  ({x.size / SAMPLE_RATE:.1f}s)")
    print(f"  speech level   : {body:6.1f} dBFS  (want louder than -45)")
    print(f"  speech-band    : {speech_share:5.1f}% of energy in 300-3400 Hz (want over 45%)")
    print(f"  zeroed samples : {zero_ratio:5.1f}%   (clean is under {ZERO_RATIO_LIMIT}%)")
    print(f"  longest gap    : {run:5.2f}s of exact silence")
    print(f"  noise floor    : {floor:6.1f} dBFS")
    print(f"  verdict        : {verdict}")
    return ok


def main() -> int:
    if len(sys.argv) > 1:
        targets = [Path(a) for a in sys.argv[1:]]
    else:
        data = ROOT / "data"
        targets = sorted(d for d in data.iterdir() if d.is_dir()) if data.is_dir() else []

    if not targets:
        print("No recordings found.")
        return 0

    results = [report(d) for d in targets]
    print()
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
