"""
====================================================================
SpeakerIdentity IA - Diarization
Tests Embedding Engine
====================================================================
"""

import math

import pytest


def test_embedding_engine_import():

    from diarization.embedding_engine import EmbeddingEngine

    assert EmbeddingEngine is not None


def test_embedding_engine_creation():

    from diarization.embedding_engine import EmbeddingEngine

    engine = EmbeddingEngine(
        device="cpu",
        auto_load=False,
    )

    assert engine is not None
    assert engine.device == "cpu"


def test_embedding_engine_not_loaded_initially():

    from diarization.embedding_engine import EmbeddingEngine

    engine = EmbeddingEngine(
        device="cpu",
        auto_load=False,
    )

    assert engine.is_loaded is False


def test_embedding_engine_statistics():

    from diarization.embedding_engine import EmbeddingEngine

    engine = EmbeddingEngine(
        device="cpu",
        auto_load=False,
    )

    stats = engine.statistics()

    assert isinstance(
        stats,
        dict,
    )

    assert stats["loaded"] is False

    assert stats["device"] == "cpu"


def test_embedding_engine_health():

    from diarization.embedding_engine import EmbeddingEngine

    engine = EmbeddingEngine(
        device="cpu",
        auto_load=False,
    )

    health = engine.health()

    assert isinstance(
        health,
        dict,
    )

    assert "status" in health


def test_pcm16_conversion():

    from diarization.embedding_engine import EmbeddingEngine

    engine = EmbeddingEngine(
        device="cpu",
        auto_load=False,
    )

    audio = b"\x00\x00" * 1600

    waveform = engine._pcm16_to_numpy(
        audio
    )

    assert waveform is not None
    assert len(waveform) == 1600


def test_cosine_similarity_identical():

    from diarization.embedding_engine import EmbeddingEngine

    embedding = [1.0, 0.0, 0.0]

    similarity = EmbeddingEngine.cosine_similarity(
        embedding,
        embedding,
    )

    assert math.isclose(
        similarity,
        1.0,
        rel_tol=1e-6,
    )


def test_cosine_similarity_different():

    from diarization.embedding_engine import EmbeddingEngine

    embedding_a = [1.0, 0.0, 0.0]

    embedding_b = [0.0, 1.0, 0.0]

    similarity = EmbeddingEngine.cosine_similarity(
        embedding_a,
        embedding_b,
    )

    assert math.isclose(
        similarity,
        0.0,
        abs_tol=1e-6,
    )


def test_cosine_distance():

    from diarization.embedding_engine import EmbeddingEngine

    embedding_a = [1.0, 0.0, 0.0]

    embedding_b = [1.0, 0.0, 0.0]

    distance = EmbeddingEngine.cosine_distance(
        embedding_a,
        embedding_b,
    )

    assert math.isclose(
        distance,
        0.0,
        abs_tol=1e-6,
    )


def test_embedding_invalid_dimensions():

    from diarization.embedding_engine import EmbeddingEngine
    from diarization.exceptions import EmbeddingError

    with pytest.raises(EmbeddingError):

        EmbeddingEngine.cosine_similarity(
            [1.0, 0.0],
            [1.0, 0.0, 0.0],
        )


def test_embedding_invalid_zero_vector():

    from diarization.embedding_engine import EmbeddingEngine
    from diarization.exceptions import EmbeddingError

    with pytest.raises(EmbeddingError):

        EmbeddingEngine.cosine_similarity(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        )