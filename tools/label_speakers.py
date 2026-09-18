"""Add speaker labels to interviews that were transcribed before labelling existed.

Reads each interview's existing transcript and its recording, works out which
side of the call every segment came from, and writes `transcript.speakers.md`
beside the transcript already there. Nothing existing is overwritten except
`transcript.json`, which gains a `speaker` and `confidence` on each segment.

Transcription is not repeated, so this takes seconds rather than an hour.

    .venv/Scripts/python.exe tools/label_speakers.py            # every interview
    .venv/Scripts/python.exe tools/label_speakers.py data/2026-...-acme
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import diarize, storage  # noqa: E402


def label(directory: Path) -> bool:
    audio = directory / "audio.webm"
    transcript = storage.read_transcript(directory)

    if transcript is None:
        print(f"{directory.name}: no transcript yet — skipped")
        return True
    if not audio.is_file():
        print(f"{directory.name}: no audio — skipped")
        return True

    segments = transcript.get("segments") or []
    if not segments:
        print(f"{directory.name}: transcript has no segments — skipped")
        return True

    labelled = diarize.label_segments(audio, segments)
    summary = diarize.summarise(labelled)

    meta = storage.read_meta(directory)
    language = transcript.get("language") or meta.get("detected_language", "")
    storage.write_transcript(directory, meta, labelled, language)
    storage.update_meta(directory, diarization=summary)

    if not summary["applied"]:
        print(f"{directory.name}: no usable channel contrast — left unlabelled")
        return False

    print(
        f"{directory.name}: {summary['segments_me']} me / "
        f"{summary['segments_phone']} phone / {summary['segments_unknown']} unknown"
        f"  -> transcript.speakers.md"
    )
    return True


def main() -> int:
    if len(sys.argv) > 1:
        targets = [Path(argument) for argument in sys.argv[1:]]
    else:
        data = ROOT / "data"
        targets = sorted(d for d in data.iterdir() if d.is_dir()) if data.is_dir() else []

    if not targets:
        print("No recordings found.")
        return 0

    results = [label(directory) for directory in targets]
    print()
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
