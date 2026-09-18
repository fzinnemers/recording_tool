# recording_tool

A small local tool for recording client interviews and turning them into
searchable transcripts. Dutch, English, or a mix of both.

Everything runs on this laptop. No account, no API key, no upload — the audio
and the transcripts never leave the machine.

## Starting it

Double-click **`start.bat`**. A browser tab opens at <http://localhost:8420>.

The first run builds a Python environment and installs the dependencies, which
takes a few minutes and needs internet. Every run after that starts in seconds.

Close the tab when you are done, and press `Ctrl+C` in the black window to stop
the server.

## Doing an interview

1. Put the phone on **speaker**, next to the laptop.
2. Click **New interview**, fill in the client name, pick the language, and tick
   the consent box.
3. On the recording screen, say something and watch the **level meter** move.
   If it does not move, pick a different microphone from the dropdown before you
   start. This is the one check worth doing every time.
4. Press **Start recording**, have the conversation, press **Stop & transcribe**.
5. Transcription runs in the background. You can close the tab; it keeps going,
   and the interview will be waiting on the home page when you come back.

Audio is written to disk every five seconds while you record, so a crash or a
closed lid costs you seconds rather than the whole interview.

## Where things are kept

Each interview gets its own folder under `data`, named by date and client:

| File | What it is |
| --- | --- |
| `audio.webm` | The recording |
| `transcript.md` | Transcript with timestamps, for reading and sharing |
| `transcript.speakers.md` | The same transcript, with each turn marked **Me** or **Phone** |
| `transcript.txt` | Plain text, handy to paste into an AI tool for a summary |
| `transcript.srt` | Subtitle format, if you ever want to caption a clip |
| `transcript.json` | Segments with exact start/end times, plus speaker and confidence |
| `meta.json` | Client, language, model, duration, and the consent record |

These are ordinary files. You can read, back up, or delete them without the app
running. Deleting an interview in the UI removes the whole folder, audio
included.

## Who said what

`transcript.speakers.md` marks every turn as **Me** or **Phone**. The plain
`transcript.md` is written exactly as before and never carries labels, so if a
label is wrong you still have the clean version sitting next to it.

It works on bandwidth, not on voices. A phone line cannot carry much above
3.4 kHz and a small speaker rolls off further still, so the far side arrives
band-limited while your own voice goes into the microphone at full width. That
is a property of the *channel*, which is why **Phone** covers everyone on the
far end however many of them there are — it separates the two sides of the
call, not individual people. The boundary is measured fresh for each recording,
so it re-calibrates to wherever the phone happened to be sitting.

A `?` after the name means the call was close and could be wrong.

**How good is it?** On a real two-minute call with three speakers it got 31 of
35 turns right. Read that as a good guess rather than a record: it is one call
in one room, and the useful figure for your own recordings is whatever you find
when you check a few.

**Where it goes wrong.** Three of those four mistakes were phone turns confidently
called *Me* — and confident, so no `?` warned about them. The likely cause is a
segment that straddles a turn: if a moment of your voice lands inside a segment
that is mostly the other side, the wideband part wins and drags the label over.
Cutting the other way, of nine turns marked `?`, only one was actually wrong, so
the question marks are cautious rather than precise. Trust the labels for the
shape of a conversation; do not lean on them for who said one specific sentence.

**When there is nothing to go on** — a mono microphone, or a phone lying against
the laptop — the two sides are not separable and no `transcript.speakers.md` is
written at all, rather than a file of confident nonsense. Three of the early test
recordings come out that way. `meta.json` records what was decided under
`diarization`.

To label interviews recorded before this existed:

```
.venv/Scripts/python.exe tools/label_speakers.py
```

It reuses the existing transcript, so it takes seconds rather than re-running
Whisper.

## The quiet side of the call

A phone on speaker sits across the desk, so the client arrives at the microphone
far quieter than your own voice — often twenty to thirty decibels down. Whisper
does not fail loudly on audio that quiet. It returns fluent Dutch that is partly
invented, or a twenty-second segment carrying six words, and what goes missing is
the half of the conversation you actually wanted.

So before transcribing, the recording is levelled: a slow envelope of the signal,
a gain that lifts each moment towards a comfortable level, capped so silence
between turns is not amplified into noise, and low-frequency rumble filtered out
first so a desk thump cannot steal the gain. Your own turns pass through
untouched — the gain never goes below one — while the far side comes up.

On one of the test interviews a two-minute passage that had transcribed as
*"Ah, ja, ja, wederpineut."* came back complete.

This runs on every transcription, costs a second or two, and needs no setting.
`meta.json` records it under `levelling`. It is a repair, not a substitute for
recording well: a phone close to the laptop still gives a better transcript than
a phone across the desk and levelled afterwards.

`transcript.speakers.md` is unaffected — the speaker labelling reads the
recording as it was captured, because it tells the two sides apart by exactly
the difference in level and bandwidth that levelling is there to remove.

## Which model to pick

`medium` is the default and is right for almost everything. Reckon on the
transcription taking roughly one to two times the length of the interview, so a
45-minute call is done in something like 45 to 90 minutes. It runs in the
background, so you can close the tab and get on with something else.

Switch to `large-v3` when the Dutch matters and you are not in a hurry — it is
noticeably better on names, jargon and accents, but two to three times slower.
`tiny` and `small` are there for a quick check, not for real work.

The first time you use a model it downloads once (1–3 GB) and needs internet.
`medium` is already downloaded. After that everything works offline.

## Consent

The consent checkbox is not decoration. You are recording an identifiable person,
so tell them at the start of the call that you are recording and transcribing,
and that it stays on your laptop. The tick and its timestamp are stored in
`meta.json` as your record that you asked.

## If something goes wrong

**The level meter does not move.** Chrome may have picked the wrong input. Try
another entry in the microphone dropdown. If the dropdown says "No microphone
access", click the padlock in the address bar, allow the microphone, and reload.

**An interview is stuck on "transcribing".** Open it and press **Transcribe
again**. If the server was closed mid-transcription, restarting `start.bat`
re-queues it automatically.

**The transcript is poor.** Usually the phone was too far from the laptop.
Move it closer and keep it off desks that rattle. Re-running with `large-v3`
via **Transcribe again** also helps, and does not need a re-recording.

**The client's answers are garbled or half-missing.** Interviews transcribed
before 2026-09-16 were not levelled (see *The quiet side of the call*). Press
**Transcribe again** and they will be, from the same recording.

**Only your own voice comes through; the client is missing.** This was a real
bug, fixed on 2026-08-31. Chrome switches on echo cancellation, noise
suppression and auto gain control by default. All three are built to isolate the
person sitting at the laptop and throw away everything else — and a phone on
speakerphone is exactly what they throw away: far-field, band-limited and
reverberant. They did not merely quieten the client, they gated that side to
*digital silence*.

The app now asks for raw, unprocessed capture, and the recording screen says
`raw capture, no filtering` under the level meter when Chrome honoured that.

**Chrome is only half of it.** A laptop *microphone array* (this machine has a
Realtek one) is beamformed by its driver: the capsules are combined into a lobe
aimed at whoever sits in front of the screen, and sound from other directions is
cancelled. That happens below the browser, so no web page can switch it off.
Turn it off in Windows: Settings → System → Sound → Input → Microphone Array →
**Audio enhancements: Off**, plus any "Voice Focus" or "Studio Effects".

If a phone on speakerphone beside the laptop still reads `too faint` on the
meter after that, the array is the problem and the reliable answer is a cheap
external USB microphone placed next to the phone. A single-capsule mic has no
beam to steer and no off-axis rejection.

To check a recording after the fact:

```
.venv/Scripts/python.exe tools/check_audio.py
```

It reports the share of *mathematically zero* samples in each interview. A real
microphone in a real room never produces exact zeros, so anything above a couple
of percent means audio was filtered out rather than just recorded quietly. The
two recordings made before the fix measured 16.2% and 14.8%.

Gated audio cannot be repaired afterwards — the sound was never written to the
file. Those interviews have to be redone.

## Development

```
.venv/Scripts/python.exe -m pytest tests -q
```

The design document is in `docs/superpowers/specs/`.
