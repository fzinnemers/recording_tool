"""Lifting the quiet side of a speakerphone call to a level Whisper can read.

A phone on speaker reaches the microphone from across the desk, so the far side
lands perhaps twenty to thirty decibels below your own voice. Whisper's encoder
does not fail loudly on audio that quiet -- it returns fluent Dutch that is
partly invented, or a twenty-second segment carrying six words, and the answers
you actually wanted are the ones missing. Recording louder is the real fix; this
is what can be done afterwards for calls already on disk.

The gain follows the recording rather than being set once: a slow envelope of
the signal, a boost that brings each moment up towards a target, and a ceiling
so silence between turns is not amplified into noise. Because the envelope is
measured over a third of a second, your own turns pass through untouched at a
gain of one while the far side is lifted, which is exactly the asymmetry the
recording has.

Rumble is removed first. Desk thumps and mains hum carry real energy but no
speech, and an envelope that measures them turns the gain *down* over precisely
the passages that needed it most.

Deliberately free of web and ML imports so it can be tested on its own. The
original recording is what `diarize.py` must keep reading: it tells the two
sides apart by their difference in level and bandwidth, and this stage exists to
erase that difference.
"""

from __future__ import annotations

import numpy as np

RUMBLE_HZ = 120.0        # below the lowest voice; only thumps and hum live here
ENVELOPE_SECONDS = 0.3   # long enough to ride syllables, short enough to track a turn
TARGET_RMS = 0.12        # level the encoder is comfortable with
MAX_GAIN = 60.0          # ~36 dB, enough for a phone across a desk
DRIVE = 1.2              # soft-clip knee, so a boosted peak bends instead of tearing
SILENCE = 1e-6           # keeps the quietest frames from dividing by zero


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    """Centred box average, in one pass rather than one per output sample.

    A direct convolution here would be minutes of arithmetic on a
    twenty-minute recording; a running sum is linear in the length of the
    signal. Edges repeat the first and last sample, so the envelope does not
    dive towards zero and invent a burst of gain at the very start.
    """
    if width < 2:
        return values.astype(np.float64)

    before = width // 2
    after = width - before - 1
    padded = np.concatenate(
        (np.full(before, values[0]), values, np.full(after, values[-1]))
    )
    running = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    return (running[width:] - running[:-width]) / width


def _without_rumble(samples: np.ndarray, sr: int) -> np.ndarray:
    """Drop everything below RUMBLE_HZ by subtracting the local mean.

    Twice, because once is not enough: a single pass leaves a 30 Hz thump only
    20 dB down, which still buries speech recorded thirty times quieter and
    puts the envelope right back where it started. A second pass takes it to
    40 dB for one more walk along the signal. What it costs is a gentle couple
    of decibels of lift around 200 Hz, which no transcript has ever minded.
    """
    width = int(round(sr / RUMBLE_HZ))
    if width < 2:
        return samples.astype(np.float64)

    voice = samples - _moving_average(samples, width)
    return voice - _moving_average(voice, width)


def normalise(samples: np.ndarray, sr: int) -> np.ndarray:
    """Even out the two sides of a call. Never attenuates, never reorders.

    Sample timing is untouched, so timestamps taken from the result still line
    up with the original recording.
    """
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    window = int(round(sr * ENVELOPE_SECONDS))
    if len(audio) < max(window, 2):
        return audio

    voice = _without_rumble(audio, sr)
    envelope = np.sqrt(np.maximum(_moving_average(voice * voice, window), 0.0))

    # Clipped at one below, so a loud near side is left exactly as it was.
    gain = np.clip(TARGET_RMS / (envelope + SILENCE), 1.0, MAX_GAIN)
    # Smoothed, or the gain steps on every syllable and the result pumps.
    gain = _moving_average(gain, max(window // 3, 2))

    return (np.tanh(voice * gain * DRIVE) / DRIVE).astype(np.float32)
