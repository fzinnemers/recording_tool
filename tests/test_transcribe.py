"""The transcription worker's two optional stages.

Levelling the audio and labelling the speakers are both bonuses on top of the
transcript, never preconditions for it, so the interesting case for each is
what happens when it cannot be done at all.
"""

from app import transcribe

SEGMENTS = [
    {"start": 0.0, "end": 1.0, "text": "Goedemiddag."},
    {"start": 1.0, "end": 2.0, "text": "Dag."},
]


def test_labelling_failure_still_yields_the_transcript(tmp_path):
    segments, diarization = transcribe._label_speakers(tmp_path / "gone.webm", SEGMENTS)
    assert segments == SEGMENTS
    assert diarization["applied"] is False
    assert diarization["error"]


def test_labelling_failure_does_not_invent_speaker_fields(tmp_path):
    segments, _ = transcribe._label_speakers(tmp_path / "gone.webm", SEGMENTS)
    assert all("speaker" not in segment for segment in segments)


def test_unreadable_audio_falls_back_to_letting_whisper_open_it(tmp_path):
    missing = tmp_path / "gone.webm"
    prepared, levelling = transcribe._prepared_audio(missing)

    assert prepared == str(missing)
    assert levelling["applied"] is False
    assert levelling["error"]


def test_a_real_recording_is_levelled_into_samples(tmp_path):
    """Whisper is handed audio, not a path, whenever levelling worked."""
    import numpy as np

    recording = tmp_path / "audio.wav"
    _write_wav(recording, (0.01 * np.sin(
        2 * np.pi * 220 * np.arange(transcribe.SAMPLE_RATE) / transcribe.SAMPLE_RATE
    )).astype(np.float32))

    prepared, levelling = transcribe._prepared_audio(recording)

    assert levelling == {"applied": True, "error": None}
    assert isinstance(prepared, np.ndarray)
    assert prepared.dtype == np.float32
    assert len(prepared) == transcribe.SAMPLE_RATE


def _write_wav(path, samples):
    import struct
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(transcribe.SAMPLE_RATE)
        scaled = (samples * 32767).astype("<i2")
        handle.writeframes(struct.pack(f"<{len(scaled)}h", *scaled.tolist()))
