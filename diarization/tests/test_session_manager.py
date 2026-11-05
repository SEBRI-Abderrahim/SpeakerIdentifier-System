"""
====================================================================
SpeakerIdentity IA - Diarization
Tests Session Manager
====================================================================
"""

import pytest


def test_session_manager_import():

    from diarization.session_manager import SessionManager

    assert SessionManager is not None


def test_session_manager_creation():

    from diarization.session_manager import SessionManager

    manager = SessionManager()

    assert manager is not None


def test_session_manager_statistics():

    from diarization.session_manager import SessionManager

    manager = SessionManager()

    if hasattr(manager, "statistics"):

        result = manager.statistics()

        assert isinstance(
            result,
            dict,
        )