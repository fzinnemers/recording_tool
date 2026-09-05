"""HTTP surface for the interview recorder. Routing only — logic lives in
storage.py and transcribe.py."""

from __future__ import annotations

import contextlib
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from . import storage, transcribe

STATIC_DIR = Path(__file__).resolve().parent / "static"
HOST = "127.0.0.1"
PORT = 8420

DOWNLOADS = {
    "md": ("transcript.md", "text/markdown; charset=utf-8"),
    "txt": ("transcript.txt", "text/plain; charset=utf-8"),
    "srt": ("transcript.srt", "application/x-subrip; charset=utf-8"),
    "json": ("transcript.json", "application/json"),
}


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    storage.DATA_DIR.mkdir(parents=True, exist_ok=True)
    transcribe.resume_interrupted()
    yield


app = FastAPI(title="record", lifespan=lifespan)


def _dir_or_404(session_id: str) -> Path:
    try:
        directory = storage.session_dir(session_id)
    except ValueError:
        raise HTTPException(400, "Invalid session id") from None
    if not (directory / "meta.json").is_file():
        raise HTTPException(404, "No such interview")
    return directory


# A cached page silently reinstates old capture settings, which is impossible to
# tell apart from a fix that did not work. Never let these two be cached.
NO_CACHE = {"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"}


def _page(name: str) -> HTMLResponse:
    return HTMLResponse(
        (STATIC_DIR / name).read_text(encoding="utf-8"), headers=NO_CACHE
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return _page("index.html")


@app.get("/diagnose", response_class=HTMLResponse)
async def diagnose() -> HTMLResponse:
    return _page("diagnose.html")


@app.get("/api/config")
async def config() -> dict:
    return {
        "models": transcribe.MODELS,
        "default_model": transcribe.DEFAULT_MODEL,
        "languages": storage.LANGUAGE_LABELS,
    }


@app.get("/api/sessions")
async def list_sessions() -> dict:
    return {"sessions": storage.list_sessions(), "queued": transcribe.queue_depth()}


@app.post("/api/sessions")
async def create_session(payload: dict = Body(...)) -> dict:
    try:
        return storage.create_session(
            client=str(payload.get("client", "")),
            language=str(payload.get("language", "auto")),
            model=str(payload.get("model", transcribe.DEFAULT_MODEL)),
            consent=bool(payload.get("consent", False)),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    directory = _dir_or_404(session_id)
    return {
        "meta": storage.read_meta(directory),
        "transcript": storage.read_transcript(directory),
    }


@app.post("/api/sessions/{session_id}/chunk")
async def upload_chunk(session_id: str, request: Request) -> dict:
    directory = _dir_or_404(session_id)
    data = await request.body()
    if not data:
        return {"bytes": 0}
    total = storage.append_chunk(directory, data)
    storage.update_meta(directory, bytes=total)
    return {"bytes": total}


@app.post("/api/sessions/{session_id}/finish")
async def finish_session(session_id: str, payload: dict = Body(default={})) -> dict:
    directory = _dir_or_404(session_id)
    meta = storage.update_meta(
        directory,
        duration=round(float(payload.get("duration", 0.0)), 2),
        capture=payload.get("capture"),
        status="transcribing",
        progress=0.0,
    )
    transcribe.enqueue(session_id)
    return meta


@app.post("/api/sessions/{session_id}/retranscribe")
async def retranscribe(session_id: str, payload: dict = Body(default={})) -> dict:
    directory = _dir_or_404(session_id)
    fields = {"status": "transcribing", "progress": 0.0, "error": None}
    if payload.get("model"):
        fields["model"] = str(payload["model"])
    meta = storage.update_meta(directory, **fields)
    transcribe.enqueue(session_id)
    return meta


@app.get("/api/sessions/{session_id}/audio")
async def get_audio(session_id: str) -> FileResponse:
    directory = _dir_or_404(session_id)
    audio = directory / "audio.webm"
    if not audio.is_file():
        raise HTTPException(404, "No audio recorded")
    return FileResponse(audio, media_type="audio/webm")


@app.get("/api/sessions/{session_id}/download/{fmt}")
async def download(session_id: str, fmt: str) -> FileResponse:
    directory = _dir_or_404(session_id)
    if fmt not in DOWNLOADS:
        raise HTTPException(404, "Unknown format")
    name, media_type = DOWNLOADS[fmt]
    path = directory / name
    if not path.is_file():
        raise HTTPException(404, "Transcript not ready yet")
    return FileResponse(path, media_type=media_type, filename=f"{session_id}.{fmt}")


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    _dir_or_404(session_id)
    storage.delete_session(session_id)
    return {"deleted": session_id}


def main() -> None:
    url = f"http://{HOST}:{PORT}"
    print(f"\n  record  ->  {url}\n  Press Ctrl+C to stop.\n")
    with contextlib.suppress(Exception):
        webbrowser.open(url)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
