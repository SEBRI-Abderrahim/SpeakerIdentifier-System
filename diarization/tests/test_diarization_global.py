"""
====================================================================
SpeakerIdentity IA - Global Diarization Test
====================================================================

Projet       : SpeakerIdentity IA
Module       : Diarization
Fichier      : tests/test_diarization_global.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 1.0.0
Date         : 14/08/2026

Description
------------

Test global d'intégration légère du package diarization.

Le test vérifie :

    - import du package ;
    - configuration audio ;
    - exceptions ;
    - modèles ;
    - événements ;
    - métriques ;
    - AudioProcessor ;
    - VAD ;
    - SegmentManager ;
    - EmbeddingEngine ;
    - SpeakerManager ;
    - SessionManager ;
    - Pipeline ;
    - DiarizationEngine ;
    - WebSocket ;
    - API router.

IMPORTANT
---------

Ce test ne charge pas automatiquement le modèle
SpeechBrain afin d'éviter une dépendance réseau/GPU
pendant le test structurel.

====================================================================
"""

from __future__ import annotations

import inspect
import sys


# ====================================================================
# PACKAGE
# ====================================================================

import diarization


# ====================================================================
# CONFIGURATION
# ====================================================================

from diarization.config import (
    SAMPLE_RATE,
    CHANNELS,
    SAMPLE_WIDTH,
    AUDIO_FORMAT,
    AUDIO_CODEC,
    config,
)


# ====================================================================
# MODELS
# ====================================================================

from diarization.models import (
    AudioCodec,
    AudioFormat,
    SessionStatus,
    SpeakerState,
    DiarizationEventType,
    SpeakerProfile,
    SpeakerSegment,
    DiarizationSession,
    DEFAULT_AUDIO_FORMAT,
)

# CORRECTIF : DiarizationEvent n'est pas dans models.py, il est defini
# dans events.py. L'import echouait, et pytest ne pouvait meme pas
# collecter ce fichier : le test d'integration ne s'executait donc
# jamais, ce qui explique le diarization_test_result.txt vide.

from diarization.events import DiarizationEvent


# ====================================================================
# EVENTS
# ====================================================================

from diarization.events import (
    DiarizationEvent as RuntimeDiarizationEvent,
    EventTypes,
)


# ====================================================================
# COMPONENTS
# ====================================================================

from diarization.audio_processor import (
    AudioProcessor,
)

from diarization.vad import (
    VoiceActivityDetector,
)

from diarization.segment_manager import (
    SegmentManager,
)

from diarization.embedding_engine import (
    EmbeddingEngine,
)

from diarization.speaker_manager import (
    SpeakerManager,
)

from diarization.session_manager import (
    DiarizationSessionManager,
)

from diarization.pipeline import (
    DiarizationPipeline,
)

from diarization.diarization_engine import (
    DiarizationEngine,
)

from diarization.websocket import (
    WebSocketSession,
    DiarizationWebSocket,
    create_websocket_handler,
    create_websocket_endpoint,
)


# ====================================================================
# HELPERS
# ====================================================================


def test_package_metadata():
    """
    Vérifie les métadonnées principales du package.
    """

    assert diarization.__project__ == "SpeakerIdentity IA"
    assert diarization.__module__ == "Diarization"
    assert diarization.__author__ == "SEBRI Abderrahim"
    assert diarization.__organization__ == (
        "SAM TEHNOLOGY Lab"
    )

    assert diarization.__version__


# ====================================================================
# AUDIO CONFIGURATION
# ====================================================================


def test_audio_configuration():
    """
    Vérifie le format audio interne SpeakerIdentity.
    """

    assert SAMPLE_RATE == 16000
    assert CHANNELS == 1
    assert SAMPLE_WIDTH == 2

    assert AUDIO_FORMAT is not None
    assert AUDIO_CODEC is not None

    assert config is not None

    assert DEFAULT_AUDIO_FORMAT is not None


# ====================================================================
# MODELS
# ====================================================================


def test_models_import():
    """
    Vérifie que les modèles principaux sont disponibles.
    """

    assert AudioCodec is not None
    assert AudioFormat is not None

    assert SessionStatus is not None
    assert SpeakerState is not None

    assert DiarizationEventType is not None

    assert SpeakerProfile is not None
    assert SpeakerSegment is not None

    assert DiarizationSession is not None
    assert DiarizationEvent is not None


# ====================================================================
# EVENTS
# ====================================================================


def test_events_import():
    """
    Vérifie les événements runtime.
    """

    assert RuntimeDiarizationEvent is not None
    assert EventTypes is not None

    assert hasattr(
        EventTypes,
        "SESSION_STARTED",
    )

    assert hasattr(
        EventTypes,
        "SESSION_STOPPED",
    )

    assert hasattr(
        EventTypes,
        "AUDIO_RECEIVED",
    )

    assert hasattr(
        EventTypes,
        "DIARIZATION_RESULT",
    )

    assert hasattr(
        EventTypes,
        "ERROR",
    )


# ====================================================================
# AUDIO BUFFER
# ====================================================================




# ====================================================================
# AUDIO PROCESSOR
# ====================================================================


def test_audio_processor():
    """
    Vérifie l'instanciation de AudioProcessor.
    """

    assert inspect.isclass(
        AudioProcessor
    )


# ====================================================================
# VAD
# ====================================================================


def test_vad():
    """
    Vérifie le VAD sans lancer de traitement audio lourd.
    """

    assert inspect.isclass(
        VoiceActivityDetector
    )

    vad_instance = VoiceActivityDetector()

    assert vad_instance is not None

    # CORRECTIF : le detecteur n'expose ni process() ni is_speech().
    # Ses methodes reelles sont detect(), process_frame() et
    # process_chunk(). L'assertion testait des noms inexistants.

    assert hasattr(
        vad_instance,
        "detect",
    )

    assert hasattr(
        vad_instance,
        "process_frame",
    )


# ====================================================================
# SEGMENT MANAGER
# ====================================================================


def test_segment_manager():
    """
    Vérifie le SegmentManager.
    """

    assert inspect.isclass(
        SegmentManager
    )


# ====================================================================
# EMBEDDING ENGINE
# ====================================================================


def test_embedding_engine():
    """
    Vérifie l'EmbeddingEngine sans charger le modèle.
    """

    assert inspect.isclass(
        EmbeddingEngine
    )

    engine = EmbeddingEngine(
        auto_load=False
    )

    assert engine is not None

    assert engine.model_name
    assert engine.sample_rate == SAMPLE_RATE

    assert engine.device in (
        "cpu",
        "cuda",
    )

    assert engine.is_loaded is False


# ====================================================================
# SPEAKER MEMORY
# ====================================================================




# ====================================================================
# SPEAKER MANAGER
# ====================================================================


def test_speaker_manager():
    """
    Vérifie le SpeakerManager.
    """

    assert inspect.isclass(
        SpeakerManager
    )


# ====================================================================
# SESSION MANAGER
# ====================================================================


def test_session_manager():
    """
    Vérifie le gestionnaire de sessions.
    """

    assert inspect.isclass(
        DiarizationSessionManager
    )


# ====================================================================
# PIPELINE
# ====================================================================


def test_pipeline():
    """
    Vérifie le pipeline.
    """

    assert inspect.isclass(
        DiarizationPipeline
    )


# ====================================================================
# DIARIZATION ENGINE
# ====================================================================


def test_diarization_engine():
    """
    Vérifie le moteur principal.
    """

    assert inspect.isclass(
        DiarizationEngine
    )


# ====================================================================
# WEBSOCKET SESSION
# ====================================================================


def test_websocket_session():
    """
    Vérifie l'état d'une session WebSocket.
    """

    state = WebSocketSession(
        session_id="test_session",
        connected_at=0.0,
    )

    assert state.session_id == "test_session"

    assert state.audio_chunks == 0
    assert state.audio_bytes == 0

    assert state.messages_received == 0
    assert state.messages_sent == 0

    assert state.closed is False

    data = state.to_dict()

    assert isinstance(
        data,
        dict,
    )

    assert data["session_id"] == (
        "test_session"
    )


# ====================================================================
# WEBSOCKET MANAGER
# ====================================================================


def test_websocket_manager():
    """
    Vérifie la création du gestionnaire WebSocket.

    Aucun WebSocket réel n'est ouvert.
    """

    class DummyEngine:
        pass

    handler = DiarizationWebSocket(
        DummyEngine()
    )

    assert handler is not None

    assert handler.engine is not None

    assert handler.max_message_bytes > 0

    assert isinstance(
        handler.health(),
        dict,
    )

    assert isinstance(
        handler.statistics(),
        dict,
    )


# ====================================================================
# WEBSOCKET FACTORIES
# ====================================================================


def test_websocket_factories():
    """
    Vérifie les factories WebSocket.
    """

    class DummyEngine:
        pass

    handler = create_websocket_handler(
        DummyEngine()
    )

    assert isinstance(
        handler,
        DiarizationWebSocket,
    )

    endpoint = create_websocket_endpoint(
        DummyEngine()
    )

    assert callable(endpoint)

    assert hasattr(
        endpoint,
        "handler",
    )

    assert isinstance(
        endpoint.handler,
        DiarizationWebSocket,
    )


# ====================================================================
# PACKAGE EXPORTS
# ====================================================================


def test_public_exports():
    """
    Vérifie que les principaux symboles publics
    sont correctement exposés par diarization.
    """

    expected_exports = [
        "VoiceActivityDetector",
        "VADResult",
        "VADSegment",
        "VADStatistics",
        "vad",
        "voice_activity_detector",

        "EmbeddingResult",
        "EmbeddingEngine",
        "create_embedding_engine",

        "WebSocketSession",
        "DiarizationWebSocket",
        "create_websocket_handler",
        "create_websocket_endpoint",

        "DiarizationEngine",
        "DiarizationPipeline",

        "SpeakerManager",

        "DiarizationSessionManager",

        "AudioProcessor",
        "SegmentManager",
    ]

    for name in expected_exports:

        assert hasattr(
            diarization,
            name,
        ), (
            f"Missing public export: {name}"
        )


# ====================================================================
# PCM16 BASIC VALIDATION
# ====================================================================


def test_pcm16_basic_validation():
    """
    Vérifie une validation PCM16 minimale
    au niveau de l'EmbeddingEngine.

    Aucun modèle n'est chargé.
    """

    engine = EmbeddingEngine(
        auto_load=False
    )

    # 100 ms à 16 kHz en PCM16 mono.
    audio = bytes(
        1600 * SAMPLE_WIDTH
    )

    waveform = engine._pcm16_to_numpy(
        audio
    )

    assert waveform is not None
    assert len(waveform) == 1600


# ====================================================================
# COSINE COMPARISON
# ====================================================================


def test_embedding_cosine():
    """
    Vérifie les fonctions mathématiques de comparaison
    des embeddings sans utiliser le modèle.
    """

    embedding_a = (
        1.0,
        0.0,
        0.0,
    )

    embedding_b = (
        1.0,
        0.0,
        0.0,
    )

    similarity = (
        EmbeddingEngine.cosine_similarity(
            embedding_a,
            embedding_b,
        )
    )

    distance = (
        EmbeddingEngine.cosine_distance(
            embedding_a,
            embedding_b,
        )
    )

    comparison = (
        EmbeddingEngine.compare(
            embedding_a,
            embedding_b,
        )
    )

    assert similarity == 1.0
    assert distance == 0.0

    assert comparison["similarity"] == 1.0
    assert comparison["distance"] == 0.0


# ====================================================================
# HEALTH / METRICS
# ====================================================================


def test_package_health_objects():
    """
    Vérifie la disponibilité des composants health et metrics.
    """

    assert hasattr(
        diarization,
        "DiarizationHealth",
    )

    assert hasattr(
        diarization,
        "DiarizationMetrics",
    )

    assert hasattr(
        diarization,
        "MetricsSnapshot",
    )

    assert hasattr(
        diarization,
        "metrics",
    )


# ====================================================================
# MAIN
# ====================================================================


if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SpeakerIdentity IA - GLOBAL DIARIZATION TEST")
    print("=" * 70)
    print()

    print(
        "Python :",
        sys.version,
    )

    print(
        "Package :",
        diarization.__version__,
    )

    print(
        "Project :",
        diarization.__project__,
    )

    print()
    print(
        "Pour exécuter les tests :"
    )

    print(
        "    python -m pytest "
        "tests/test_diarization_global.py -v"
    )

    print()