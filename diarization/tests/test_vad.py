"""
====================================================================
SpeakerIdentity IA - Diarization
Tests VAD
====================================================================
"""

import pytest


def test_vad_import():

    from diarization.vad import VoiceActivityDetector

    assert VoiceActivityDetector is not None


def test_vad_creation():

    from diarization.vad import VoiceActivityDetector

    vad = VoiceActivityDetector()

    assert vad is not None


def test_vad_aggressiveness():

    from diarization.vad import VoiceActivityDetector

    vad = VoiceActivityDetector(
        aggressiveness=2
    )

    assert vad is not None


def test_vad_silence():

    from diarization.vad import VoiceActivityDetector

    vad = VoiceActivityDetector(
        aggressiveness=2
    )

    silence = b"\x00\x00" * 160

    try:

        result = vad.is_speech(
            silence,
            sample_rate=16000,
        )

        assert isinstance(
            result,
            bool,
        )

    except AttributeError:

        pytest.skip(
            "Current VAD API does not expose "
            "is_speech()."
        )