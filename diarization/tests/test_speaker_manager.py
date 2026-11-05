"""
====================================================================
SpeakerIdentity IA - Diarization
Tests Speaker Manager
====================================================================
"""

import pytest


def test_speaker_manager_import():

    from diarization.speaker_manager import SpeakerManager

    assert SpeakerManager is not None


def test_speaker_manager_creation():

    from diarization.speaker_manager import SpeakerManager

    manager = SpeakerManager()

    assert manager is not None


def test_speaker_manager_statistics():

    from diarization.speaker_manager import SpeakerManager

    manager = SpeakerManager()

    if hasattr(manager, "statistics"):

        result = manager.statistics()

        assert isinstance(
            result,
            dict,
        )