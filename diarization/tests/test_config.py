"""
====================================================================
SpeakerIdentity IA - Diarization
Tests Configuration
====================================================================
"""

import importlib

import pytest


def test_config_import():

    config = importlib.import_module(
        "diarization.config"
    )

    assert config is not None


def test_audio_configuration():

    from diarization.config import (
        SAMPLE_RATE,
        CHANNELS,
        SAMPLE_WIDTH,
        SAMPLE_FORMAT,
        AUDIO_DTYPE,
    )

    assert SAMPLE_RATE == 16000
    assert CHANNELS == 1
    assert SAMPLE_WIDTH == 2
    assert SAMPLE_FORMAT == "PCM16"
    assert AUDIO_DTYPE == "<i2"


def test_vad_configuration():

    from diarization.config import (
        VAD_FRAME_DURATION_MS,
        VAD_AGGRESSIVENESS,
    )

    assert VAD_FRAME_DURATION_MS in {
        10,
        20,
        30,
    }

    assert VAD_AGGRESSIVENESS in {
        0,
        1,
        2,
        3,
    }


def test_embedding_configuration():

    from diarization.config import (
        EMBEDDING_ENABLED,
        EMBEDDING_MODEL,
        EMBEDDING_DIMENSION,
        EMBEDDING_SAMPLE_RATE,
        EMBEDDING_MIN_AUDIO_MS,
        EMBEDDING_MAX_AUDIO_MS,
    )

    assert isinstance(
        EMBEDDING_ENABLED,
        bool,
    )

    assert EMBEDDING_MODEL.strip()

    assert EMBEDDING_DIMENSION > 0

    assert EMBEDDING_SAMPLE_RATE == 16000

    assert (
        EMBEDDING_MIN_AUDIO_MS
        <= EMBEDDING_MAX_AUDIO_MS
    )


def test_device_configuration():

    from diarization.config import (
        DEVICE,
        USE_GPU,
        GPU_DEVICE_INDEX,
        CPU_THREADS,
    )

    assert DEVICE in {
        "cpu",
        "cuda",
        "auto",
    }

    assert isinstance(
        USE_GPU,
        bool,
    )

    assert GPU_DEVICE_INDEX >= 0

    assert CPU_THREADS > 0


def test_websocket_configuration():

    from diarization.config import (
        WEBSOCKET_ENABLED,
        WEBSOCKET_HOST,
        WEBSOCKET_PORT,
        WEBSOCKET_PATH,
    )

    assert isinstance(
        WEBSOCKET_ENABLED,
        bool,
    )

    assert WEBSOCKET_HOST

    assert 1 <= WEBSOCKET_PORT <= 65535

    assert WEBSOCKET_PATH.startswith("/")


def test_api_configuration():

    from diarization.config import (
        API_HOST,
        API_PORT,
        API_PREFIX,
    )

    assert API_HOST

    assert 1 <= API_PORT <= 65535

    assert API_PREFIX.startswith("/")