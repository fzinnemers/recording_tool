"""Session directories, metadata and transcript rendering.

Deliberately free of web and ML imports so it can be tested on its own.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# meta.json is read by the web request thread and written by the worker thread.
_meta_lock = threading.Lock()

VALID_LANGUAGES = {"auto", "nl", "en"}
LANGUAGE_LABELS = {"auto": "Auto-detect", "nl": "Nederlands", "en": "English"}

# Speaker labels describe the channel a voice arrived on, not who was talking:
# everyone on the far end of the call is "Phone", however many of them there
# were. Below this confidence the label is printed with a question mark, since
# a visible doubt is worth more than a tidy guess.
SPEAKER_NAMES = {"me": "Me", "phone": "Phone"}
UNCERTAIN_BELOW = 0.35


def slugify(name: str) -> str:
    """Turn a client name into something safe for a Windows directory name."""
    normalised = unicodedata.normalize("NFKD", name)
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_only).strip("-").lower()
    return slug[:40] or "interview"


def session_dir(session_id: str) -> Path:
    """Resolve a session id to its directory, refusing anything path-like."""
    if "/" in session_id or "\\" in session_id or session_id in {"", ".", ".."}:
        raise ValueError(f"invalid session id: {session_id!r}")
    directory = (DATA_DIR / session_id).resolve()
    if directory.parent != DATA_DIR.resolve():
        raise ValueError(f"invalid session id: {session_id!r}")
    return directory


def create_session(client: str, language: str, model: str, consent: bool) -> dict:
    if not client.strip():
        raise ValueError("client name is required")
    if language not in VALID_LANGUAGES:
        raise ValueError(f"unknown language: {language!r}")
    if not consent:
        raise ValueError("consent is required before recording")

    now = datetime.now()
    base = f"{now:%Y-%m-%d_%H%M}_{slugify(client)}"
    directory = DATA_DIR / base
    suffix = 2
    while directory.exists():
        directory = DATA_DIR / f"{base}-{suffix}"
        suffix += 1
    directory.mkdir(parents=True)

    meta = {
        "id": directory.name,
        "client": client.strip(),
        "language": language,
        "model": model,
        "consent": True,
        "consent_at": now.astimezone(timezone.utc).isoformat(),
        "created_at": now.astimezone(timezone.utc).isoformat(),
        "status": "recording",
        "progress": 0.0,
        "duration": 0.0,
        "bytes": 0,
        "error": None,
        "notes": "",
    }
    write_meta(directory, meta)
    return meta


def read_meta(directory: Path) -> dict:
    with _meta_lock:
        return json.loads((directory / "meta.json").read_text(encoding="utf-8"))


def write_meta(directory: Path, meta: dict) -> None:
    with _meta_lock:
        tmp = directory / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(directory / "meta.json")


def update_meta(directory: Path, **fields) -> dict:
    """Read-modify-write meta.json atomically with respect to other threads."""
    with _meta_lock:
        path = directory / "meta.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.update(fields)
        tmp = directory / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return meta


def list_sessions() -> list[dict]:
    sessions = []
    for directory in sorted(DATA_DIR.iterdir(), reverse=True) if DATA_DIR.exists() else []:
        if not (directory / "meta.json").is_file():
            continue
        try:
            sessions.append(read_meta(directory))
        except (OSError, json.JSONDecodeError):
            continue
    return sessions


def delete_session(session_id: str) -> None:
    shutil.rmtree(session_dir(session_id))


def append_chunk(directory: Path, data: bytes) -> int:
    """Append one MediaRecorder chunk to the session audio; return total bytes."""
    audio = directory / "audio.webm"
    with audio.open("ab") as handle:
        handle.write(data)
    return audio.stat().st_size


# --- transcript rendering ---------------------------------------------------


def _clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _srt_clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def render_txt(segments: list[dict]) -> str:
    return "\n".join(segment["text"].strip() for segment in segments) + "\n"


def render_markdown(meta: dict, segments: list[dict], language: str) -> str:
    created = meta.get("created_at", "")[:16].replace("T", " ")
    lines = [
        f"# Interview — {meta.get('client', 'unknown')}",
        "",
        f"- **Date:** {created} UTC",
        f"- **Duration:** {_clock(meta.get('duration', 0.0))}",
        f"- **Detected language:** {language}",
        f"- **Model:** {meta.get('model', '')}",
        f"- **Consent recorded:** {'yes' if meta.get('consent') else 'no'}",
        "",
        "## Transcript",
        "",
    ]
    for segment in segments:
        lines.append(f"**[{_clock(segment['start'])}]** {segment['text'].strip()}")
        lines.append("")
    return "\n".join(lines)


def has_speaker_labels(segments: list[dict]) -> bool:
    return any(segment.get("speaker") in SPEAKER_NAMES for segment in segments)


def _speaker_name(segment: dict) -> str:
    speaker = segment.get("speaker")
    if speaker not in SPEAKER_NAMES:
        return "?"
    name = SPEAKER_NAMES[speaker]
    return f"{name}?" if segment.get("confidence", 0.0) < UNCERTAIN_BELOW else name


def render_speaker_markdown(meta: dict, segments: list[dict], language: str) -> str:
    """The transcript with each turn attributed to a side of the call."""
    created = meta.get("created_at", "")[:16].replace("T", " ")
    lines = [
        f"# Interview — {meta.get('client', 'unknown')} (speaker-labelled)",
        "",
        f"- **Date:** {created} UTC",
        f"- **Duration:** {_clock(meta.get('duration', 0.0))}",
        f"- **Detected language:** {language}",
        f"- **Model:** {meta.get('model', '')}",
        "",
        "> **Me** is the microphone in the room, **Phone** is everyone on the far",
        "> end of the call — it separates the two sides, not individual voices.",
        "> A `?` marks a turn the labelling was unsure about. Labels are a good",
        "> guess, not a transcript of who spoke; `transcript.md` is the unlabelled",
        "> original if one looks wrong.",
        "",
        "## Transcript",
        "",
    ]
    for segment in segments:
        lines.append(
            f"**[{_clock(segment['start'])}] {_speaker_name(segment)}:** "
            f"{segment['text'].strip()}"
        )
        lines.append("")
    return "\n".join(lines)


def render_srt(segments: list[dict]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n"
            f"{_srt_clock(segment['start'])} --> {_srt_clock(segment['end'])}\n"
            f"{segment['text'].strip()}\n"
        )
    return "\n".join(blocks)


def write_transcript(directory: Path, meta: dict, segments: list[dict], language: str) -> None:
    payload = {"language": language, "segments": segments}
    (directory / "transcript.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "transcript.txt").write_text(render_txt(segments), encoding="utf-8")
    (directory / "transcript.md").write_text(
        render_markdown(meta, segments, language), encoding="utf-8"
    )
    (directory / "transcript.srt").write_text(render_srt(segments), encoding="utf-8")

    # The labelled transcript sits beside the plain one rather than replacing
    # it, so a wrong label costs nothing. When no side could be worked out,
    # any file from an earlier run is cleared rather than left to go stale.
    labelled = directory / "transcript.speakers.md"
    if has_speaker_labels(segments):
        labelled.write_text(
            render_speaker_markdown(meta, segments, language), encoding="utf-8"
        )
    else:
        labelled.unlink(missing_ok=True)


def read_transcript(directory: Path) -> dict | None:
    path = directory / "transcript.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
