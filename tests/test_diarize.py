"""Channel attribution: telling your voice from the phone's.

The two sides of a speakerphone recording differ in bandwidth, not in timbre:
a telephone codec cannot reproduce anything above roughly 3.4 kHz, while a
voice going straight into the microphone can. These tests synthesise both
kinds of signal rather than lean on recordings, because `data/` is gitignored.
"""

import numpy as np
import pytest

from app import diarize

SR = 48000

# Long-term speech spectrum measured off a real recording, as dB per band
# relative to the 200-500 Hz band. Synthesising against this keeps the fixture
# honest: a flat-noise fixture would make the job far easier than reality.
SPEECH_SHAPE = [
    (200, 0.0), (500, -5.0), (1000, -12.0), (2000, -18.0), (3000, -20.0),
    (4500, -25.0), (6000, -28.0), (8000, -32.0), (12000, -40.0), (24000, -48.0),
]


def synth_voice(seconds, band_limit=None, hf_trim_db=0.0, seed=0):
    """Speech-shaped noise. `band_limit` emulates a telephone codec."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    knots_f = np.array([f for f, _ in SPEECH_SHAPE], dtype=float)
    knots_db = np.array([d for _, d in SPEECH_SHAPE], dtype=float)
    shape_db = np.interp(np.clip(freqs, knots_f[0], knots_f[-1]), knots_f, knots_db)
    shape = 10 ** (shape_db / 20.0)
    shape[freqs < 150] = 0.0
    if hf_trim_db:
        shape[freqs >= 5000] *= 10 ** (-hf_trim_db / 20.0)
    if band_limit is not None:
        # A codec floor, not digital silence: the room mic still hears the room.
        # -14 dB matches the gap measured on a real recording; a deeper floor
        # would make this fixture easier than the job actually is.
        shape[freqs > band_limit] *= 10 ** (-14.0 / 20.0)
    spectrum = np.fft.rfft(rng.normal(0, 1, n)) * shape
    out = np.fft.irfft(spectrum, n)
    return (out / (np.abs(out).max() + 1e-12) * 0.5).astype(np.float32)


def build(spec):
    """spec: list of (kind, seconds). Returns (samples, segments)."""
    parts, segments, t = [], [], 0.0
    for i, (kind, seconds) in enumerate(spec):
        if kind == "room":
            part = synth_voice(seconds, seed=i)
        elif kind == "phone":
            part = synth_voice(seconds, band_limit=3400, seed=i)
        elif kind == "silence":
            part = np.zeros(int(seconds * SR), dtype=np.float32)
        else:
            raise ValueError(kind)
        parts.append(part)
        segments.append({"start": t, "end": t + seconds, "text": f"{kind} {i}"})
        t += seconds
    return np.concatenate(parts), segments


ALTERNATING = [("room", 2.0), ("phone", 2.0)] * 5


def test_alternating_channels_are_told_apart():
    samples, segments = build(ALTERNATING)
    labelled = diarize.classify_segments(samples, SR, segments)
    got = [s["speaker"] for s in labelled]
    assert got == ["me", "phone"] * 5


def test_uniform_recording_yields_no_labels_rather_than_a_guess():
    """No channel contrast means the method does not apply. Say so."""
    samples, segments = build([("phone", 2.0)] * 10)
    labelled = diarize.classify_segments(samples, SR, segments)
    assert {s["speaker"] for s in labelled} == {"unknown"}


def test_confident_segments_report_high_confidence():
    samples, segments = build(ALTERNATING)
    labelled = diarize.classify_segments(samples, SR, segments)
    assert all(s["confidence"] > 0.5 for s in labelled)


def test_segment_too_short_to_measure_is_unknown():
    spec = [("room", 2.0), ("phone", 2.0)] * 4 + [("room", 0.02)]
    samples, segments = build(spec)
    labelled = diarize.classify_segments(samples, SR, segments)
    assert labelled[-1]["speaker"] == "unknown"
    assert labelled[-1]["confidence"] == 0.0


def test_silent_segment_is_unknown():
    spec = [("room", 2.0), ("phone", 2.0)] * 4 + [("silence", 2.0)]
    samples, segments = build(spec)
    labelled = diarize.classify_segments(samples, SR, segments)
    assert labelled[-1]["speaker"] == "unknown"


def test_original_segment_fields_are_preserved():
    samples, segments = build(ALTERNATING)
    labelled = diarize.classify_segments(samples, SR, segments)
    assert [s["text"] for s in labelled] == [s["text"] for s in segments]
    assert [s["start"] for s in labelled] == [s["start"] for s in segments]
    assert segments[0].get("speaker") is None, "input must not be mutated"


def test_broadband_click_does_not_turn_a_phone_segment_into_a_room_one():
    """A door slam is broadband but is not speech; it must not carry a label."""
    samples, segments = build(ALTERNATING)
    click_at = int((segments[1]["start"] + 0.5) * SR)
    samples[click_at:click_at + 60] += 8.0     # far above any speech level
    labelled = diarize.classify_segments(samples, SR, segments)
    assert labelled[1]["speaker"] == "phone"


class TestSmoothIsolatedFlips:
    """One wobbly segment inside a confident run is a measurement error, not a
    turn. Speakers talk in runs; a single-segment interjection with almost no
    margin is the shape of noise."""

    def test_low_margin_flip_between_confident_neighbours_is_corrected(self):
        labels = ["me", "me", "phone", "me", "me"]
        margins = [6.0, 5.0, 0.4, 5.5, 6.0]
        assert diarize.smooth_isolated_flips(labels, margins) == ["me"] * 5

    def test_confident_flip_is_left_alone(self):
        labels = ["me", "me", "phone", "me", "me"]
        margins = [6.0, 5.0, 7.0, 5.5, 6.0]
        assert diarize.smooth_isolated_flips(labels, margins) == labels

    def test_flip_beside_an_unconfident_neighbour_is_left_alone(self):
        labels = ["me", "me", "phone", "me", "me"]
        margins = [6.0, 0.5, 0.4, 5.5, 6.0]
        assert diarize.smooth_isolated_flips(labels, margins) == labels

    def test_a_real_two_segment_turn_is_left_alone(self):
        labels = ["me", "phone", "phone", "me"]
        margins = [6.0, 0.4, 0.4, 6.0]
        assert diarize.smooth_isolated_flips(labels, margins) == labels

    def test_unknown_segments_are_never_rewritten(self):
        labels = ["me", "me", "unknown", "me", "me"]
        margins = [6.0, 5.0, 0.0, 5.5, 6.0]
        assert diarize.smooth_isolated_flips(labels, margins)[2] == "unknown"
