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
    segments_iter, info = model.transcribe(
        str(audio),
        language=None if language == "auto" else language,
        vad_filter=True,
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
