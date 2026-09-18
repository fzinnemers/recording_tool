"""Telling your voice apart from the phone's, in a speakerphone recording.

The two sides differ in *bandwidth*, not in timbre. A telephone codec cannot
carry much above 3.4 kHz, and a small phone speaker rolls off further still, so
the far side reaches the microphone band-limited. Your own voice goes straight
in, full width. That difference is a property of the channel rather than of the
person, which is why this can label a phone side shared by two people without
ever telling their voices apart -- and why it says "phone" rather than a name.

Measured as an *upper envelope*: a codec imposes a hard ceiling on
high-frequency energy, while your voice merely dips below that ceiling now and
then, on a vowel. Averaging conflates the two and ends up measuring vowels; a
high percentile measures the ceiling, which is the thing that actually differs.

The boundary is learned per recording, never hardcoded, so it re-calibrates to
wherever the phone happened to sit. When a recording has no usable contrast --
a mono microphone, a phone pressed against the laptop -- the method does not
apply, and it reports `unknown` instead of guessing.

Deliberately free of web and ML imports so it can be tested on its own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

WIN = 1024
HOP = 512

HF_BAND = (5000.0, 12000.0)   # dead above the codec ceiling, alive for a room mic
REF_BAND = (200.0, 1500.0)    # body of speech, present on both sides
ENVELOPE_PERCENTILE = 80      # the ceiling, not the average

SPEECH_FLOOR_DB = 22.0        # frames this far below the loud end are not speech
MIN_FRAMES = 4                # too little audio to measure a ceiling
MIN_CLUSTER_GAP_DB = 6.0      # below this the two clusters are one population
WOBBLE_MARGIN_DB = 1.5        # within this of the boundary, the call is soft
CONFIDENT_MARGIN_DB = 2.5     # beyond this a neighbour can vouch for a wobble
FULL_CONFIDENCE_DB = 6.0      # margin at which confidence reaches 1.0

ME = "me"
PHONE = "phone"
UNKNOWN = "unknown"


def _frame_features(samples: np.ndarray, sr: int):
    """Per-frame high-band-to-body ratio in dB, plus the body level itself."""
    if len(samples) < WIN:
        empty = np.empty(0)
        return empty, empty, empty

    count = 1 + (len(samples) - WIN) // HOP
    index = np.arange(WIN)[None, :] + HOP * np.arange(count)[:, None]
    power = np.abs(np.fft.rfft(samples[index] * np.hanning(WIN), axis=1)) ** 2
    freqs = np.fft.rfftfreq(WIN, 1.0 / sr)

    hf_hi = min(HF_BAND[1], sr / 2.0)
    hf = power[:, (freqs >= HF_BAND[0]) & (freqs < hf_hi)].sum(axis=1) + 1e-14
    ref = power[:, (freqs >= REF_BAND[0]) & (freqs < REF_BAND[1])].sum(axis=1) + 1e-14

    times = (np.arange(count) * HOP + WIN / 2.0) / sr
    ref_db = 10.0 * np.log10(ref)
    return times, 10.0 * np.log10(hf) - ref_db, ref_db


def _speech_frames(ref_db: np.ndarray) -> np.ndarray:
    """Frames carrying speech. Silence has no channel to measure.

    An earlier version also rejected frames whose highs were too loud to be
    speech, on the theory that a click would inflate the envelope. Measured
    against a real call it made things worse, not better -- 28/35 against
    30/35 -- because dropping those frames drags the learned boundary down and
    tips borderline phone segments the wrong way, turning soft errors into
    confident ones. The percentile already shrugs off brief transients, so the
    rejection bought nothing and cost accuracy. It is gone deliberately.
    """
    return ref_db > (np.percentile(ref_db, 95) - SPEECH_FLOOR_DB)


def _kmeans_1d(values: np.ndarray, iterations: int = 100):
    """Two clusters on one dimension. Seeded from the deciles so the result
    does not depend on a random draw."""
    low, high = np.percentile(values, 10), np.percentile(values, 90)
    for _ in range(iterations):
        upper = np.abs(values - high) < np.abs(values - low)
        if not upper.any() or upper.all():
            break
        low_new, high_new = values[~upper].mean(), values[upper].mean()
        if np.isclose(low_new, low) and np.isclose(high_new, high):
            break
        low, high = low_new, high_new
    return low, high


def smooth_isolated_flips(labels: list[str], margins: list[float]) -> list[str]:
    """Correct a lone wobble inside a confident run.

    People speak in runs. A single segment labelled against both its
    neighbours, with almost no margin of its own while they are confident, is
    far more likely a measurement wobble than a real one-segment turn. A
    genuine short turn keeps its label, because its neighbours differ from each
    other rather than agreeing.
    """
    smoothed = list(labels)
    for i in range(1, len(labels) - 1):
        before, here, after = labels[i - 1], labels[i], labels[i + 1]
        if here == UNKNOWN or before == UNKNOWN:
            continue
        if before != after or before == here:
            continue
        if margins[i] >= WOBBLE_MARGIN_DB:
            continue
        if min(margins[i - 1], margins[i + 1]) >= CONFIDENT_MARGIN_DB:
            smoothed[i] = before
    return smoothed


def classify_segments(samples, sr: int, segments: list[dict]) -> list[dict]:
    """Add `speaker` and `confidence` to a copy of each transcript segment."""
    labelled = [dict(segment) for segment in segments]
    values = np.full(len(segments), np.nan)

    if sr >= 2 * HF_BAND[0]:
        audio = np.asarray(samples, dtype=np.float64)
        times, hf_db, ref_db = _frame_features(audio, sr)
        if len(times):
            speech = _speech_frames(ref_db)
            for i, segment in enumerate(segments):
                inside = speech & (times >= segment["start"]) & (times < segment["end"])
                if inside.sum() >= MIN_FRAMES:
                    values[i] = np.percentile(hf_db[inside], ENVELOPE_PERCENTILE)

    labels = [UNKNOWN] * len(segments)
    margins = [0.0] * len(segments)
    usable = ~np.isnan(values)

    if usable.sum() >= 2:
        low, high = _kmeans_1d(values[usable])
        if high - low >= MIN_CLUSTER_GAP_DB:
            boundary = (low + high) / 2.0
            for i in np.flatnonzero(usable):
                margin = values[i] - boundary
                labels[i] = ME if margin > 0 else PHONE
                margins[i] = abs(float(margin))
            labels = smooth_isolated_flips(labels, margins)

    for segment, label, margin in zip(labelled, labels, margins):
        segment["speaker"] = label
        segment["confidence"] = (
            0.0 if label == UNKNOWN
            else round(min(1.0, margin / FULL_CONFIDENCE_DB), 3)
        )
    return labelled


def summarise(labelled: list[dict]) -> dict:
    """A record of what the classifier did, for meta.json."""
    counts = {ME: 0, PHONE: 0, UNKNOWN: 0}
    for segment in labelled:
        counts[segment.get("speaker", UNKNOWN)] += 1
    return {
        "method": "hf-ceiling",
        "applied": counts[ME] > 0 or counts[PHONE] > 0,
        "segments_me": counts[ME],
        "segments_phone": counts[PHONE],
        "segments_unknown": counts[UNKNOWN],
    }


def _decode(path: Path):
    """Decode any container ffmpeg understands to mono float, at its own rate."""
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.audio[0]
        sr = stream.codec_context.sample_rate
        chunks = []
        for frame in container.decode(stream):
            array = frame.to_ndarray()
            channels = frame.layout.nb_channels
            if array.ndim == 1:
                array = array.reshape(-1, channels).T
            elif array.shape[0] == 1 and channels > 1:
                array = array.reshape(-1, channels).T
            if np.issubdtype(array.dtype, np.integer):
                array = array.astype(np.float32) / np.iinfo(array.dtype).max
            chunks.append(array.astype(np.float32))
    finally:
        container.close()

    if not chunks:
        return np.zeros(0, dtype=np.float32), sr
    return np.concatenate(chunks, axis=1).mean(axis=0), sr


def label_segments(audio_path, segments: list[dict]) -> list[dict]:
    """Label a transcript's segments from the recording they came from."""
    samples, sr = _decode(Path(audio_path))
    return classify_segments(samples, sr, segments)
