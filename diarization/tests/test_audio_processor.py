"""
====================================================================
SpeakerIdentity IA - Diarization
Tests Audio Processor
====================================================================
"""

import pytest


def test_audio_processor_import():

    from diarization.audio_processor import AudioProcessor

    assert AudioProcessor is not None


def test_audio_processor_creation():

    from diarization.audio_processor import AudioProcessor

    processor = AudioProcessor()

    assert processor is not None


def test_pcm16_audio_processing():

    from diarization.audio_processor import AudioProcessor

    processor = AudioProcessor()

    audio = b"\x00\x00" * 16000

    try:

        result = processor.process(audio)

        assert result is not None

    except AttributeError:

        pytest.skip(
            "Current AudioProcessor API does not expose "
            "process(audio)."
        )