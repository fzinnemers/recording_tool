# Interview Recorder & Transcriber — Design

**Date:** 2026-08-31
**Status:** Approved, implementing

## Purpose

Record client-discovery interviews and turn them into searchable text.
Interviews are conducted by phone on speakerphone next to the laptop, so the
laptop microphone captures both sides as a single stream. Conversations may be
in Dutch, English, or a mix of both.

## Constraints

- **Privacy first.** Recordings are of real, identifiable people. Audio and
  transcripts never leave the laptop; no cloud API, no account, no telemetry.
- **Windows 11**, Python 3.13 already present. No system `ffmpeg` installed and
  none required.
- Must be launchable by someone who is about to start a phone call — one
  double-click, no terminal gymnastics.

## Architecture

A local web app: a Python process serving a single-page UI on
`http://localhost:8420`.

| Concern | Choice | Rationale |
| --- | --- | --- |
| Audio capture | Browser `MediaRecorder` (Opus in WebM) | Avoids PortAudio/PyAudio build problems on Windows; the browser supplies a device picker and permission flow |
| Backend | FastAPI + uvicorn | Minimal routing surface, static file serving, background work |
| Transcription | `faster-whisper` on CPU, `int8` | Offline, free, good Dutch. Its bundled PyAV decodes WebM/Opus, removing the ffmpeg dependency |
| Storage | Plain directories under `data/` | Transcripts readable and greppable without the app running |
| Frontend | One vanilla HTML/CSS/JS file | No build step, no npm |

### Modules

- `app/storage.py` — session directories, metadata read/write, transcript
  rendering (md / txt / srt). No web or ML imports; unit-testable on its own.
- `app/transcribe.py` — a single background worker thread with a job queue and
  a model cache. Reports progress by writing into the session metadata.
- `app/server.py` — HTTP routes only; delegates to the two modules above.
- `app/static/index.html` — the UI.

Single worker thread is deliberate: two Whisper models loaded at once would
exhaust RAM on a laptop, and interviews are transcribed one at a time anyway.

### Data flow

1. `POST /api/sessions` creates `data/<YYYY-MM-DD_HHMM>_<client-slug>/` with
   `meta.json` (client, language, consent, model, status).
2. The browser records with a 5-second timeslice. Each chunk is `POST`ed to
   `/api/sessions/{id}/chunk` and appended to `audio.webm`. Uploads are
   serialised client-side so chunks land in order — the first chunk carries the
   WebM header, later chunks are clusters, and concatenation yields a decodable
   stream. A crash mid-interview costs at most five seconds.
3. `POST /api/sessions/{id}/finish` records the duration and enqueues a
   transcription job.
4. The worker runs Whisper with `vad_filter=True`, writing `transcript.json`,
   `transcript.md`, `transcript.txt` and `transcript.srt`, updating
   `status` and `progress` in `meta.json` as segments arrive.
5. The UI polls `GET /api/sessions/{id}` until `status` is `done` or `error`.

### Session states

`recording` → `transcribing` → `done`, with `error` reachable from
`transcribing`. On startup the server re-queues anything left in `transcribing`
and demotes a stranded `recording` to `error` with an explanatory message, so a
crashed run never leaves a session that polls forever.

## User interface

Four screens in one page:

- **Home** — list of interviews (client, date, duration, status), plus a "New
  interview" button.
- **New** — client name, language (`Auto` / `Nederlands` / `English`), Whisper
  model, and a consent checkbox that must be ticked. The consent flag and its
  timestamp are stored in `meta.json` as a record that permission was obtained.
- **Recording** — microphone selector, large Start/Stop control, elapsed timer,
  and a live input-level meter so the operator can confirm the phone's
  speakerphone is actually being picked up before the conversation starts.
- **Detail** — status/progress, audio player, timestamped transcript, a search
  box that filters segments, and download links.

## Out of scope for v1

- **Speaker diarization.** One room microphone plus a speakerphone gives poor
  channel separation, and `pyannote` would add PyTorch and a HuggingFace token
  for an unreliable result. Timestamps plus context are enough to follow who is
  speaking.
- **Summarisation.** Transcripts are plain text files; an LLM can be pointed at
  them afterwards.
- **Multi-user / networked access.** The server binds to `127.0.0.1` only.

## Testing

- Unit tests over `storage.py`: slug generation, session creation, metadata
  round-tripping, and SRT/Markdown rendering (including timestamp formatting at
  the hour boundary).
- Manual acceptance: record ~30 seconds of mixed Dutch and English through the
  laptop microphone and confirm the transcript is coherent and the downloads
  render.
