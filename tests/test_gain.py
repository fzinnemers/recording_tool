"""Levelling the two sides of a speakerphone call.

The property that matters is the asymmetry: the far side comes up, the near
side stays where it is, and nothing moves in time.
"""

import numpy as np

from app import gain

SR = 16000


def _speech(seconds: float, amplitude: float, freq: float = 220.0) -> np.ndarray:
    """A voiced-sounding tone: a fundamental in the body of speech, plus a harmonic."""
    t = np.arange(int(SR * seconds)) / SR
    tone = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
    return (amplitude * tone / 1.5).astype(np.float32)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2)))


def test_a_quiet_recording_is_lifted_towards_the_target():
    lifted = gain.normalise(_speech(4.0, 0.01), SR)
    assert _rms(lifted) > 0.05


def test_a_loud_recording_is_left_roughly_alone():
    loud = _speech(4.0, 0.3)
    assert _rms(gain.normalise(loud, SR)) > 0.8 * _rms(loud)


def test_the_quiet_side_gains_far_more_than_the_loud_side():
    """The whole point: a phone across the desk catches up with the near voice."""
    near, far = _speech(3.0, 0.3), _speech(3.0, 0.01)
    levelled = gain.normalise(np.concatenate([near, far]), SR)

    # Ignore the half second around the seam, where the envelope spans both.
    edge = int(0.5 * SR)
    near_out, far_out = levelled[: len(near) - edge], levelled[len(near) + edge :]

    assert _rms(far_out) / _rms(far) > 5 * (_rms(near_out) / _rms(near))
    assert _rms(far_out) > 0.3 * _rms(near_out)


def test_silence_is_not_amplified_without_limit():
    quiet = (1e-7 * np.random.default_rng(0).standard_normal(SR * 3)).astype(np.float32)
    assert _rms(gain.normalise(quiet, SR)) <= gain.MAX_GAIN * _rms(quiet)


def test_rumble_is_removed_rather_than_amplified():
    """A desk thump has energy but no speech; left in, it steals the gain."""
    t = np.arange(SR * 3) / SR
    rumble = (0.3 * np.sin(2 * np.pi * 30 * t)).astype(np.float32)
    levelled = gain.normalise(rumble + _speech(3.0, 0.01), SR)

    spectrum = np.abs(np.fft.rfft(levelled.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(levelled), 1 / SR)
    low = spectrum[(freqs > 20) & (freqs < 50)].max()
    voice = spectrum[(freqs > 200) & (freqs < 240)].max()
    assert voice > low


def test_timing_and_dtype_survive():
    """Timestamps are read off this audio, and the VAD only accepts float32."""
    samples = _speech(2.0, 0.05)
    levelled = gain.normalise(samples, SR)
    assert levelled.shape == samples.shape
    assert levelled.dtype == np.float32


def test_a_clip_shorter_than_the_envelope_is_passed_through():
    samples = _speech(0.1, 0.05)
    assert np.array_equal(gain.normalise(samples, SR), samples)


def test_the_running_sum_matches_a_plain_convolution():
    """Guards the one-pass average that makes this affordable on a long call."""
    values = np.random.default_rng(1).standard_normal(500)
    width = 21
    fast = gain._moving_average(values, width)
    slow = np.convolve(values, np.ones(width) / width, mode="same")

    inside = slice(width, -width)  # the edges are padded differently by design
    assert np.allclose(fast[inside], slow[inside])
