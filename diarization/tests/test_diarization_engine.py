"""
====================================================================
SpeakerIdentity IA - Diarization
Tests Diarization Engine
====================================================================
"""

import pytest


def test_diarization_engine_import():

    from diarization.diarization_engine import DiarizationEngine

    assert DiarizationEngine is not None


def test_diarization_engine_creation():

    from diarization.diarization_engine import DiarizationEngine

    engine = DiarizationEngine()

    assert engine is not None


def test_diarization_engine_health():

    from diarization.diarization_engine import DiarizationEngine

    engine = DiarizationEngine()

    if hasattr(engine, "health"):

        result = engine.health()

        assert isinstance(
            result,
            dict,
        )

        assert "status" in result