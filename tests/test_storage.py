import json

import pytest

from app import storage


@pytest.fixture(autouse=True)
def temp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path / "data")
    storage.DATA_DIR.mkdir()
    return storage.DATA_DIR


def test_slugify_strips_accents_and_punctuation():
    assert storage.slugify("Jansen Bouw B.V. — Peter") == "jansen-bouw-b-v-peter"
    assert storage.slugify("Café Zurüch") == "cafe-zuruch"


def test_slugify_falls_back_when_nothing_survives():
    assert storage.slugify("***") == "interview"


def test_create_session_writes_meta():
    meta = storage.create_session("Acme BV", "nl", "medium", consent=True)
    directory = storage.session_dir(meta["id"])
    assert directory.is_dir()
    on_disk = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    assert on_disk["client"] == "Acme BV"
    assert on_disk["status"] == "recording"
    assert on_disk["consent"] is True
    assert on_disk["consent_at"]


@pytest.mark.parametrize(
    "client,language,consent",
    [("", "nl", True), ("Acme", "fr", True), ("Acme", "nl", False)],
)
def test_create_session_rejects_bad_input(client, language, consent):
    with pytest.raises(ValueError):
        storage.create_session(client, language, "medium", consent=consent)


def test_two_sessions_same_minute_get_distinct_directories():
    first = storage.create_session("Acme", "auto", "medium", consent=True)
    second = storage.create_session("Acme", "auto", "medium", consent=True)
    assert first["id"] != second["id"]


def test_session_dir_rejects_traversal():
    BS = chr(92)
    for bad in ["../secrets", "a/b", "a" + BS + "b", "", "..", ".." + BS + ".."]:
        with pytest.raises(ValueError):
            storage.session_dir(bad)


def test_append_chunk_accumulates():
    meta = storage.create_session("Acme", "auto", "medium", consent=True)
    directory = storage.session_dir(meta["id"])
    assert storage.append_chunk(directory, b"header") == 6
    assert storage.append_chunk(directory, b"-cluster") == 14
    assert (directory / "audio.webm").read_bytes() == b"header-cluster"


def test_update_meta_merges_and_persists():
    meta = storage.create_session("Acme", "auto", "medium", consent=True)
    directory = storage.session_dir(meta["id"])
    storage.update_meta(directory, status="done", duration=12.5)
    reloaded = storage.read_meta(directory)
    assert reloaded["status"] == "done"
    assert reloaded["duration"] == 12.5
    assert reloaded["client"] == "Acme"


def test_list_sessions_skips_stray_directories():
    storage.create_session("Acme", "auto", "medium", consent=True)
    (storage.DATA_DIR / "not-a-session").mkdir()
    assert [s["client"] for s in storage.list_sessions()] == ["Acme"]


SEGMENTS = [
    {"start": 0.0, "end": 2.5, "text": "Goedemiddag, bedankt voor uw tijd."},
    {"start": 3601.25, "end": 3605.0, "text": "So what would you change?"},
]


def test_srt_pads_hours_and_milliseconds():
    srt = storage.render_srt(SEGMENTS)
    assert "00:00:00,000 --> 00:00:02,500" in srt
    assert "01:00:01,250 --> 01:00:05,000" in srt
    assert srt.startswith("1\n")


def test_clock_only_shows_hours_when_needed():
    assert storage._clock(65) == "01:05"
    assert storage._clock(3661) == "1:01:01"


def test_transcript_round_trip_writes_all_formats():
    meta = storage.create_session("Acme", "nl", "medium", consent=True)
    directory = storage.session_dir(meta["id"])
    storage.write_transcript(directory, meta, SEGMENTS, "nl")

    assert storage.read_transcript(directory)["segments"] == SEGMENTS
    assert storage.read_transcript(directory)["language"] == "nl"

    txt = (directory / "transcript.txt").read_text(encoding="utf-8")
    assert "Goedemiddag, bedankt voor uw tijd." in txt

    markdown = (directory / "transcript.md").read_text(encoding="utf-8")
    assert "# Interview — Acme" in markdown
    assert "**[1:00:01]**" in markdown
    assert "Consent recorded:** yes" in markdown


def test_read_transcript_returns_none_before_transcription():
    meta = storage.create_session("Acme", "nl", "medium", consent=True)
    assert storage.read_transcript(storage.session_dir(meta["id"])) is None


LABELLED = [
    {"start": 0.0, "end": 2.5, "text": "Goedemiddag, bedankt voor uw tijd.",
     "speaker": "me", "confidence": 0.9},
    {"start": 3601.25, "end": 3605.0, "text": "So what would you change?",
     "speaker": "phone", "confidence": 0.15},
]


def test_labelling_never_changes_the_plain_transcripts():
    """The unlabelled transcript is the fallback when a label is wrong, so it
    must be byte-identical whether or not speakers were worked out."""
    meta = storage.create_session("Acme", "nl", "medium", consent=True)
    assert storage.render_markdown(meta, LABELLED, "nl") == storage.render_markdown(
        meta, SEGMENTS, "nl"
    )
    assert storage.render_txt(LABELLED) == storage.render_txt(SEGMENTS)
    assert storage.render_srt(LABELLED) == storage.render_srt(SEGMENTS)


def test_speaker_transcript_names_both_sides():
    meta = storage.create_session("Acme", "nl", "medium", consent=True)
    rendered = storage.render_speaker_markdown(meta, LABELLED, "nl")
    assert "**[00:00] Me:**" in rendered
    assert "Goedemiddag, bedankt voor uw tijd." in rendered


def test_speaker_transcript_flags_an_uncertain_label():
    meta = storage.create_session("Acme", "nl", "medium", consent=True)
    rendered = storage.render_speaker_markdown(meta, LABELLED, "nl")
    assert "**[1:00:01] Phone?:**" in rendered


def test_write_transcript_adds_the_labelled_file():
    meta = storage.create_session("Acme", "nl", "medium", consent=True)
    directory = storage.session_dir(meta["id"])
    storage.write_transcript(directory, meta, LABELLED, "nl")
    assert (directory / "transcript.speakers.md").is_file()
    assert storage.read_transcript(directory)["segments"][0]["speaker"] == "me"


def test_no_labelled_file_when_speakers_are_unknown():
    """A recording with no channel contrast gets no labelled transcript at all,
    rather than a file full of 'unknown'."""
    meta = storage.create_session("Acme", "nl", "medium", consent=True)
    directory = storage.session_dir(meta["id"])
    unknown = [dict(s, speaker="unknown", confidence=0.0) for s in SEGMENTS]
    storage.write_transcript(directory, meta, unknown, "nl")
    assert not (directory / "transcript.speakers.md").exists()


def test_labelled_file_is_removed_when_relabelling_finds_nothing():
    """Re-transcribing must not leave a stale labelled transcript behind."""
    meta = storage.create_session("Acme", "nl", "medium", consent=True)
    directory = storage.session_dir(meta["id"])
    storage.write_transcript(directory, meta, LABELLED, "nl")
    unknown = [dict(s, speaker="unknown", confidence=0.0) for s in SEGMENTS]
    storage.write_transcript(directory, meta, unknown, "nl")
    assert not (directory / "transcript.speakers.md").exists()


def test_write_transcript_still_works_without_speaker_fields():
    meta = storage.create_session("Acme", "nl", "medium", consent=True)
    directory = storage.session_dir(meta["id"])
    storage.write_transcript(directory, meta, SEGMENTS, "nl")
    assert not (directory / "transcript.speakers.md").exists()
    assert (directory / "transcript.md").is_file()
