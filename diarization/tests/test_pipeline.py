"""
====================================================================
SpeakerIdentity IA - Diarization
Tests Pipeline
====================================================================
"""

import pytest


def test_pipeline_import():

    from diarization.pipeline import Pipeline

    assert Pipeline is not None


def test_pipeline_creation():

    from diarization.pipeline import Pipeline

    pipeline = Pipeline()

    assert pipeline is not None


def test_pipeline_health():

    from diarization.pipeline import Pipeline

    pipeline = Pipeline()

    if hasattr(pipeline, "health"):

        result = pipeline.health()

        assert isinstance(
            result,
            dict,
        )