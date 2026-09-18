"""Background transcription worker built on faster-whisper.

One worker thread processes one interview at a time. Two Whisper models loaded
concurrently would exhaust RAM on a laptop, and interviews arrive one by one.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from pathlib import Path

from . import storage

MODELS = ["tiny", "base", "small", "medium", "large-v3"]
DEFAULT_MODEL = "medium"

SAMPLE_RATE = 16000   # what Whisper resamples to anyway; decoding here costs nothing
VAD_THRESHOLD = 0.3   # the 0.5 default drops far-side speech that is soft even boosted
SPEECH_PAD_MS = 300   # keep the run-up to a quiet word rather than clipping into it

_jobs: "queue.Queue[str]" = queue.Queue()
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()
_model_cache: dict[str, object] = {}


def enqueue(session_id: str) -> None:
    """Queue a session for transcription, starting the worker if needed."""
    _ensure_worker()
    _jobs.put(session_id)


def queue_depth() -> int:
    return _jobs.qsize()


def _ensure_worker() -> None:
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run, name="transcriber", daemon=True)
            _worker.start()


def _run() -> None:
    while True:
        session_id = _jobs.get()
        try:
            _transcribe(session_id)
        except Exception:  # noqa: BLE001 - worker must never die on one bad job
            _fail(session_id, traceback.format_exc(limit=3))
        finally:
            _jobs.task_done()


def _fail(session_id: str, message: str) -> None:
    try:
        storage.update_meta(
            storage.session_dir(session_id), status="error", error=message.strip()[-800:]
        )
    except OSError:
        pass


def _load_model(name: str):
    if name not in _model_cache:
        from faster_whisper import WhisperModel

        # int8 on CPU is the practical trade-off on a laptop: roughly 2x faster
        # than float32 with no meaningful accuracy loss for interview audio.
        _model_cache.clear()
        _model_cache[name] = WhisperModel(name, device="cpu", compute_type="int8")
    return _model_cache[name]


def _prepared_audio(audio: Path):
    """Decode the recording and even out the two sides of the call.

    Never fatal. On any failure Whisper opens the file itself, which is what it
    did before this stage existed: a worse transcript of a quiet phone, not a
    missing one.
    """
    try:
        from faster_whisper.audio import decode_audio

        from . import gain

        samples = decode_audio(str(audio), sampling_rate=SAMPLE_RATE)
        return gain.normalise(samples, SAMPLE_RATE), {"applied": True, "error": None}
    except Exception as exc:  # noqa: BLE001 - a quiet transcript beats no transcript
        return str(audio), {
            "applied": False,
            "error": f"{type(exc).__name__}: {exc}".strip()[:300],
        }


def _label_speakers(audio: Path, segments: list[dict]) -> tuple[list[dict], dict]:
    """Attribute each segment to a side of the call.

    Never fatal. An unlabelled transcript is still a good transcript, so a
    failure here costs the labels and nothing else.

    Reads the recording as it was captured, never the levelled audio: it tells
    the two sides apart by how they differ in level and bandwidth, which is the
    very difference `gain.normalise` is there to remove.
    """
    from . import diarize

    try:
        labelled = diarize.label_segments(audio, segments)
    except Exception as exc:  # noqa: BLE001 - labels are a bonus, not the job
        return segments, {
            "applied": False,
            "error": f"{type(exc).__name__}: {exc}".strip()[:300],
        }
    return labelled, diarize.summarise(labelled)


def _transcribe(session_id: str) -> None:
    directory = storage.session_dir(session_id)
    meta = storage.read_meta(directory)
    audio = directory / "audio.webm"

    if not audio.is_file() or audio.stat().st_size == 0:
        storage.update_meta(
            directory, status="error", error="No audio was recorded for this session."
        )
        return

    storage.update_meta(directory, status="transcribing", progress=0.0, error=None)

    model_name = meta.get("model") or DEFAULT_MODEL
    model = _load_model(model_name)

    language = meta.get("language", "auto")
    prepared, levelling = _prepared_audio(audio)
    segments_iter, info = model.transcribe(
        prepared,
        language=None if language == "auto" else language,
        vad_filter=True,
        vad_parameters={"threshold": VAD_THRESHOLD, "speech_pad_ms": SPEECH_PAD_MS},
        beam_size=5,
    )

    total = float(getattr(info, "duration", 0.0) or 0.0)
    detected = getattr(info, "language", None) or language

    segments: list[dict] = []
    last_write = 0.0
    for segment in segments_iter:
        segments.append(
            {
                "start": round(float(segment.start), 2),
                "end": round(float(segment.end), 2),
                "text": segment.text.strip(),
            }
        )
        now = time.monotonic()
        if now - last_write > 2.0:
            last_write = now
            progress = min(0.99, float(segment.end) / total) if total else 0.0
            storage.update_meta(directory, progress=round(progress, 3))

    segments, diarization = _label_speakers(audio, segments)

    meta = storage.read_meta(directory)
    if total and not meta.get("duration"):
        meta = storage.update_meta(directory, duration=round(total, 2))

    storage.write_transcript(directory, meta, segments, detected)
    storage.update_meta(
        directory,
        status="done",
        progress=1.0,
        detected_language=detected,
        segment_count=len(segments),
        levelling=levelling,
        diarization=diarization,
        error=None,
    )


def resume_interrupted() -> None:
    """Repair sessions left mid-flight by a previous run of the server."""
    for meta in storage.list_sessions():
        if meta.get("status") == "transcribing":
            enqueue(meta["id"])
        elif meta.get("status") == "recording":
            directory = storage.session_dir(meta["id"])
            audio = directory / "audio.webm"
            if audio.is_file() and audio.stat().st_size > 0:
                enqueue(meta["id"])
            else:
                storage.update_meta(
                    directory,
                    status="error",
                    error="Recording was interrupted before any audio was saved.",
                )
