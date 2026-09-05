"""Guards the fix for the phone-audio bug.

Chrome's default echoCancellation/noiseSuppression/autoGainControl gate a
speakerphone's far-field audio to digital silence: an interview recorded with
the defaults came back 16.2% mathematically-zero samples, with the client's
side missing from the transcript entirely. The recording screen must therefore
ask for raw capture, and must never fall back to `audio: true`.
"""

import re
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parent.parent / "app" / "static" / "index.html"
PROCESSING_FLAGS = ["echoCancellation", "noiseSuppression", "autoGainControl"]


@pytest.fixture(scope="module")
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


@pytest.mark.parametrize("flag", PROCESSING_FLAGS)
def test_processing_flag_is_disabled(page, flag):
    assert re.search(rf"{flag}\s*:\s*false", page), f"{flag} must be explicitly disabled"


@pytest.mark.parametrize("flag", PROCESSING_FLAGS)
def test_processing_flag_is_never_enabled(page, flag):
    assert not re.search(rf"{flag}\s*:\s*true", page), f"{flag} must never be turned on"


def test_recording_never_requests_bare_audio_true(page):
    # `audio: true` silently reinstates every default filter.
    assert not re.search(r"audio\s*:\s*true", page)


def test_constraints_are_passed_to_getusermedia(page):
    call = re.search(r"getUserMedia\(\{\s*audio:\s*(\w+)\s*\}\)", page)
    assert call, "getUserMedia should receive a prepared constraints object"
    assert re.search(rf"{call.group(1)}\s*=\s*Object\.assign\(\{{\}},\s*RAW_AUDIO\)", page)


def test_operator_is_warned_when_chrome_ignores_the_constraints(page):
    # Chrome can refuse the constraints; silently recording a gated interview
    # is the failure mode this whole fix exists to prevent.
    assert "getSettings()" in page
    assert "still active" in page


def test_diagnostic_page_measures_both_configurations():
    diagnose = (PAGE.parent / "diagnose.html").read_text(encoding="utf-8")
    for flag in PROCESSING_FLAGS:
        assert f"{flag}: false" in diagnose
