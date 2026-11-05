"""
====================================================================
SpeakerIdentity IA - Diarization Package
====================================================================

Projet       : SpeakerIdentity IA
Module       : Diarization
Fichier      : diarization/__init__.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 3.0.0
Date         : 14/08/2026

Description
------------

Package principal du système de diarisation SpeakerIdentity IA.

Fonctionnalités principales :

    - traitement audio PCM16 ;
    - audio 16 kHz mono ;
    - buffering temps réel ;
    - prétraitement audio ;
    - VAD WebRTC + fallback RMS ;
    - segmentation audio ;
    - génération d'embeddings vocaux ;
    - identification des speakers ;
    - mémoire vocale ;
    - gestion des sessions ;
    - pipeline temps réel ;
    - WebSocket ;
    - API REST ;
    - health check ;
    - métriques ;
    - intégration avec le STT.

IMPORTANT
---------

Aucun modèle lourd n'est chargé automatiquement lors de
l'import du package.

Les composants lourds doivent être instanciés explicitement
par l'application principale.

====================================================================
"""

from __future__ import annotations


# ====================================================================
# PACKAGE METADATA
# ====================================================================

__version__ = "3.0.0"

__author__ = "SEBRI Abderrahim"

__organization__ = "SAM TEHNOLOGY Lab"

__project__ = "SpeakerIdentity IA"

__module__ = "Diarization"


# ====================================================================
# CONFIGURATION
# ====================================================================

# IMPORTANT :
#
# config.py définit :
#
#     SAMPLE_RATE
#     CHANNELS
#     SAMPLE_WIDTH
#     SAMPLE_FORMAT
#     AUDIO_DTYPE
#
# Il ne définit PAS :
#
#     AUDIO_FORMAT
#     AUDIO_CODEC
#
# On importe donc uniquement les constantes réellement présentes.

from .config import (
    PROJECT_NAME,
    MODULE_NAME,
    MODULE_VERSION,
    AUTHOR,
    ORGANIZATION,

    DEBUG,
    LOG_LEVEL,

    SAMPLE_RATE,
    CHANNELS,
    SAMPLE_WIDTH,
    SAMPLE_FORMAT,
    AUDIO_DTYPE,

    AUDIO_BUFFER_MS,
    AUDIO_BUFFER_MAX_SECONDS,

    AUDIO_NORMALIZE,
    AUDIO_TARGET_SAMPLE_RATE,
    AUDIO_TARGET_CHANNELS,
    AUDIO_TARGET_SAMPLE_WIDTH,

    VAD_ENABLED,
    VAD_AGGRESSIVENESS,
    VAD_FRAME_DURATION_MS,
    VAD_MIN_SPEECH_FRAMES,
    VAD_MIN_SILENCE_FRAMES,

    SEGMENT_MIN_DURATION_MS,
    SEGMENT_MAX_DURATION_MS,
    SEGMENT_SILENCE_DURATION_MS,

    EMBEDDING_ENABLED,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_SAMPLE_RATE,
    EMBEDDING_MIN_AUDIO_MS,
    EMBEDDING_MAX_AUDIO_MS,

    INFERENCE_BATCH_SIZE,

    DEVICE,
    USE_GPU,
    GPU_DEVICE_INDEX,
    CPU_THREADS,

    SPEAKER_SIMILARITY_THRESHOLD,

    SPEAKER_MEMORY_ENABLED,
    SPEAKER_MEMORY_MAX_SPEAKERS,
    SPEAKER_MEMORY_MAX_EMBEDDINGS_PER_SPEAKER,
    SPEAKER_PROTOTYPE_SIZE,

    SESSION_ENABLED,
    SESSION_MAX_ACTIVE,
    SESSION_TIMEOUT_SECONDS,

    PIPELINE_ENABLED,
    PIPELINE_QUEUE_SIZE,
    PIPELINE_MAX_WORKERS,

    WEBSOCKET_ENABLED,
    WEBSOCKET_HOST,
    WEBSOCKET_PORT,
    WEBSOCKET_PATH,

    API_HOST,
    API_PORT,
    API_PREFIX,

    HEALTH_ENABLED,
    HEALTH_PATH,

    METRICS_ENABLED,
)


# ====================================================================
# CONFIGURATION OBJECTS
# ====================================================================

# Certaines versions du module peuvent fournir des objets
# DiarizationConfig / config / diarization_config.
#
# Ils sont volontairement chargés de manière optionnelle afin
# que l'import principal reste compatible avec la configuration
# actuelle.

# ====================================================================
# CHARGEMENT DES SOUS-MODULES
# ====================================================================
#
# CORRECTIF : chaque bloc ci-dessous etait un
#
#     try:  from .X import (...)
#     except ImportError:  A = None ; B = None ...
#
# Une seule erreur d'import mettait donc a None TOUT le bloc, sans le
# moindre message. Concretement, `from diarization import config`
# renvoyait None parce que le bloc importait `diarization_config`, un
# nom qui n'existait pas ; et `from diarization import SpeakerMemory`
# renvoyait None a cause de deux symboles fantomes. Dix-sept noms
# publics valaient None alors que les classes correspondantes
# existaient et fonctionnaient. La panne n'apparaissait que bien plus
# loin, sous la forme d'un TypeError: 'NoneType' object is not callable.
#
# Le chargement reste tolerant, mais il n'est plus muet : chaque echec
# est enregistre dans IMPORT_ERRORS, un avertissement est emis a
# l'import, et diagnostic() donne le detail.
#
# ====================================================================

import warnings as _warnings

IMPORT_ERRORS: dict = {}


def _signaler_import(module: str, exc: BaseException) -> None:
    """Enregistre un echec d'import au lieu de l'avaler en silence."""

    IMPORT_ERRORS[module] = "%s: %s" % (type(exc).__name__, exc)


try:
    from .config import (
        DiarizationConfig,
        config,
        diarization_config,
    )
except ImportError as _exc:
    _signaler_import("config", _exc)
    DiarizationConfig = None
    # config : on NE remet PAS a None, ce nom est aussi celui du sous-module
    diarization_config = None


# ====================================================================
# DATA MODELS
# ====================================================================

try:

    from .models import (
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

except ImportError as _exc:
    _signaler_import("models", _exc)

    AudioCodec = None
    AudioFormat = None
    SessionStatus = None
    SpeakerState = None
    DiarizationEventType = None
    SpeakerProfile = None
    SpeakerSegment = None
    DiarizationSession = None
    DEFAULT_AUDIO_FORMAT = None


# ====================================================================
# EXCEPTIONS
# ====================================================================

try:

    from .exceptions import (
        DiarizationError,
        AudioFormatError,
        AudioBufferError,
        SessionError,
        SessionNotFoundError,
        SpeakerError,
        SpeakerIdentificationError,
        EmbeddingError,
        PipelineError,
        WebSocketError,
        ConfigurationError,
    )

except ImportError as _exc:
    _signaler_import("exceptions", _exc)

    class DiarizationError(Exception):
        pass

    class AudioFormatError(DiarizationError):
        pass

    class AudioBufferError(DiarizationError):
        pass

    class SessionError(DiarizationError):
        pass

    class SessionNotFoundError(SessionError):
        pass

    class SpeakerError(DiarizationError):
        pass

    class SpeakerIdentificationError(SpeakerError):
        pass

    class EmbeddingError(DiarizationError):
        pass

    class PipelineError(DiarizationError):
        pass

    class WebSocketError(DiarizationError):
        pass

    class ConfigurationError(DiarizationError):
        pass


# ====================================================================
# EVENTS
# ====================================================================

try:

    from .events import (
        DiarizationEvent,
        DiarizationEvent as RuntimeDiarizationEvent,
        EventTypes,
    )

except ImportError as _exc:
    _signaler_import("events", _exc)

    DiarizationEvent = None
    RuntimeDiarizationEvent = None
    EventTypes = None


# ====================================================================
# METRICS
# ====================================================================

try:

    from .metrics import (
        MetricsSnapshot,
        DiarizationMetrics,
        metrics,
    )

except ImportError as _exc:
    _signaler_import("metrics", _exc)

    MetricsSnapshot = None
    DiarizationMetrics = None
    # metrics : on NE remet PAS a None, ce nom est aussi celui du sous-module


# ====================================================================
# HEALTH
# ====================================================================

try:

    from .health import (
        DiarizationHealth,
    )

except ImportError as _exc:
    _signaler_import("health", _exc)

    DiarizationHealth = None


# ====================================================================
# AUDIO BUFFER
# ====================================================================

# audio_buffer.py a ete retire : aucun module ne l'importait.
# audio_processor.py fait sa propre gestion de tampon.


# ====================================================================
# AUDIO PROCESSOR
# ====================================================================

try:

    from .audio_processor import (
        AudioProcessor,
    )

except ImportError as _exc:
    _signaler_import("audio_processor", _exc)

    AudioProcessor = None


# ====================================================================
# VAD
# ====================================================================

try:

    from .vad import (
        VoiceActivityDetector,
        VADResult,
        VADSegment,
        VADStatistics,
        vad,
        voice_activity_detector,
    )

except ImportError as _exc:
    _signaler_import("vad", _exc)

    VoiceActivityDetector = None
    VADResult = None
    VADSegment = None
    VADStatistics = None
    # vad : on NE remet PAS a None, ce nom est aussi celui du sous-module
    voice_activity_detector = None


# ====================================================================
# SEGMENT MANAGER
# ====================================================================

try:

    from .segment_manager import (
        SegmentManager,
    )

except ImportError as _exc:
    _signaler_import("segment_manager", _exc)

    SegmentManager = None


# ====================================================================
# EMBEDDING ENGINE
# ====================================================================

try:

    from .embedding_engine import (
        EmbeddingResult,
        EmbeddingEngine,
        create_embedding_engine,
    )

except ImportError as _exc:
    _signaler_import("embedding_engine", _exc)

    EmbeddingResult = None
    EmbeddingEngine = None
    create_embedding_engine = None


# ====================================================================
# SPEAKER MEMORY
# ====================================================================

# speaker_memory.py a ete retire : c'etait une SECONDE implementation
# de la logique de speaker_manager.py (autre cosinus, autre seuil, autre
# format de resultat), que personne n'importait. Un commentaire y
# affirmait pourtant servir de "compatibilite utilisee par
# SpeakerManager", ce qui etait faux et pretait a confusion.


# ====================================================================
# SPEAKER MANAGER
# ====================================================================

try:

    from .speaker_manager import (
        SpeakerManager,
    )

except ImportError as _exc:
    _signaler_import("speaker_manager", _exc)

    SpeakerManager = None


# ====================================================================
# SESSION MANAGER
# ====================================================================

try:

    from .session_manager import (
        DiarizationSessionManager,
    )

except ImportError as _exc:
    _signaler_import("session_manager", _exc)

    DiarizationSessionManager = None


# ====================================================================
# PIPELINE
# ====================================================================

try:

    from .pipeline import (
        DiarizationPipeline,
    )

except ImportError as _exc:
    _signaler_import("pipeline", _exc)

    DiarizationPipeline = None


# ====================================================================
# DIARIZATION ENGINE
# ====================================================================

try:

    from .diarization_engine import (
        DiarizationEngine,
    )

except ImportError as _exc:
    _signaler_import("diarization_engine", _exc)

    DiarizationEngine = None


# ====================================================================
# WEBSOCKET
# ====================================================================

try:

    from .websocket import (
        WebSocketSession,
        DiarizationWebSocket,
        create_websocket_handler,
        create_websocket_endpoint,
    )

except ImportError as _exc:
    _signaler_import("websocket", _exc)

    WebSocketSession = None
    DiarizationWebSocket = None
    create_websocket_handler = None
    create_websocket_endpoint = None


# ====================================================================
# API
# ====================================================================

try:

    from .api import (
        create_diarization_router,
    )

except ImportError as _exc:
    _signaler_import("api", _exc)

    create_diarization_router = None


# ====================================================================
# PUBLIC API
# ====================================================================

__all__ = [

    # ---------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------

    "__version__",
    "__author__",
    "__organization__",
    "__project__",
    "__module__",

    "PROJECT_NAME",
    "MODULE_NAME",
    "MODULE_VERSION",
    "AUTHOR",
    "ORGANIZATION",

    # ---------------------------------------------------------------
    # General configuration
    # ---------------------------------------------------------------

    "DEBUG",
    "LOG_LEVEL",

    # ---------------------------------------------------------------
    # Audio configuration
    # ---------------------------------------------------------------

    "SAMPLE_RATE",
    "CHANNELS",
    "SAMPLE_WIDTH",
    "SAMPLE_FORMAT",
    "AUDIO_DTYPE",

    "AUDIO_BUFFER_MS",
    "AUDIO_BUFFER_MAX_SECONDS",

    "AUDIO_NORMALIZE",
    "AUDIO_TARGET_SAMPLE_RATE",
    "AUDIO_TARGET_CHANNELS",
    "AUDIO_TARGET_SAMPLE_WIDTH",

    # ---------------------------------------------------------------
    # VAD
    # ---------------------------------------------------------------

    "VAD_ENABLED",
    "VAD_AGGRESSIVENESS",
    "VAD_FRAME_DURATION_MS",
    "VAD_MIN_SPEECH_FRAMES",
    "VAD_MIN_SILENCE_FRAMES",

    # ---------------------------------------------------------------
    # Segmentation
    # ---------------------------------------------------------------

    "SEGMENT_MIN_DURATION_MS",
    "SEGMENT_MAX_DURATION_MS",
    "SEGMENT_SILENCE_DURATION_MS",

    # ---------------------------------------------------------------
    # Embedding
    # ---------------------------------------------------------------

    "EMBEDDING_ENABLED",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_SAMPLE_RATE",
    "EMBEDDING_MIN_AUDIO_MS",
    "EMBEDDING_MAX_AUDIO_MS",
    "INFERENCE_BATCH_SIZE",

    # ---------------------------------------------------------------
    # Device
    # ---------------------------------------------------------------

    "DEVICE",
    "USE_GPU",
    "GPU_DEVICE_INDEX",
    "CPU_THREADS",

    # ---------------------------------------------------------------
    # Speaker
    # ---------------------------------------------------------------

    "SPEAKER_SIMILARITY_THRESHOLD",

    # ---------------------------------------------------------------
    # Speaker memory
    # ---------------------------------------------------------------

    "SPEAKER_MEMORY_ENABLED",
    "SPEAKER_MEMORY_MAX_SPEAKERS",
    "SPEAKER_MEMORY_MAX_EMBEDDINGS_PER_SPEAKER",
    "SPEAKER_PROTOTYPE_SIZE",

    # ---------------------------------------------------------------
    # Session
    # ---------------------------------------------------------------

    "SESSION_ENABLED",
    "SESSION_MAX_ACTIVE",
    "SESSION_TIMEOUT_SECONDS",

    # ---------------------------------------------------------------
    # Pipeline
    # ---------------------------------------------------------------

    "PIPELINE_ENABLED",
    "PIPELINE_QUEUE_SIZE",
    "PIPELINE_MAX_WORKERS",

    # ---------------------------------------------------------------
    # WebSocket
    # ---------------------------------------------------------------

    "WEBSOCKET_ENABLED",
    "WEBSOCKET_HOST",
    "WEBSOCKET_PORT",
    "WEBSOCKET_PATH",

    # ---------------------------------------------------------------
    # API
    # ---------------------------------------------------------------

    "API_HOST",
    "API_PORT",
    "API_PREFIX",

    # ---------------------------------------------------------------
    # Health
    # ---------------------------------------------------------------

    "HEALTH_ENABLED",
    "HEALTH_PATH",

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    "METRICS_ENABLED",

    # ---------------------------------------------------------------
    # Configuration objects
    # ---------------------------------------------------------------

    "DiarizationConfig",
    "config",
    "diarization_config",

    # ---------------------------------------------------------------
    # Models
    # ---------------------------------------------------------------

    "AudioCodec",
    "AudioFormat",
    "DEFAULT_AUDIO_FORMAT",

    "SessionStatus",
    "SpeakerState",
    "DiarizationEventType",

    "SpeakerProfile",
    "SpeakerSegment",
    "DiarizationSession",
    "DiarizationEvent",

    # ---------------------------------------------------------------
    # Exceptions
    # ---------------------------------------------------------------

    "DiarizationError",
    "AudioFormatError",
    "AudioBufferError",
    "SessionError",
    "SessionNotFoundError",
    "SpeakerError",
    "SpeakerIdentificationError",
    "EmbeddingError",
    "PipelineError",
    "WebSocketError",
    "ConfigurationError",

    # ---------------------------------------------------------------
    # Events
    # ---------------------------------------------------------------

    "RuntimeDiarizationEvent",
    "EventTypes",

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    "MetricsSnapshot",
    "DiarizationMetrics",
    "metrics",

    # ---------------------------------------------------------------
    # Health
    # ---------------------------------------------------------------

    "DiarizationHealth",

    # ---------------------------------------------------------------
    # Audio processing
    # ---------------------------------------------------------------

    "AudioProcessor",

    # ---------------------------------------------------------------
    # VAD
    # ---------------------------------------------------------------

    "VoiceActivityDetector",
    "VADResult",
    "VADSegment",
    "VADStatistics",
    "vad",
    "voice_activity_detector",

    # ---------------------------------------------------------------
    # Segmentation
    # ---------------------------------------------------------------

    "SegmentManager",

    # ---------------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------------

    "EmbeddingResult",
    "EmbeddingEngine",
    "create_embedding_engine",

    # ---------------------------------------------------------------
    # Speaker memory
    # ---------------------------------------------------------------


    # ---------------------------------------------------------------
    # Speaker manager
    # ---------------------------------------------------------------

    "SpeakerManager",

    # ---------------------------------------------------------------
    # Session manager
    # ---------------------------------------------------------------

    "DiarizationSessionManager",

    # ---------------------------------------------------------------
    # Pipeline
    # ---------------------------------------------------------------

    "DiarizationPipeline",

    # ---------------------------------------------------------------
    # Main engine
    # ---------------------------------------------------------------

    "DiarizationEngine",

    # ---------------------------------------------------------------
    # WebSocket
    # ---------------------------------------------------------------

    "WebSocketSession",
    "DiarizationWebSocket",
    "create_websocket_handler",
    "create_websocket_endpoint",

    # ---------------------------------------------------------------
    # API
    # ---------------------------------------------------------------

    "create_diarization_router",
]

# ====================================================================
# ALIAS PUBLICS
# ====================================================================
#
# Les tests et le serveur de demonstration utilisent les noms courts
# Pipeline et SessionManager. Les classes reelles s'appellent
# DiarizationPipeline et DiarizationSessionManager. On expose les deux
# plutot que de renommer les classes, pour ne rien casser ailleurs.
#
# ====================================================================

Pipeline = DiarizationPipeline

SessionManager = DiarizationSessionManager

__all__ = list(__all__) + [
    "Pipeline",
    "SessionManager",
    "IMPORT_ERRORS",
    "diagnostic",
]


# ====================================================================
# DIAGNOSTIC
# ====================================================================

def diagnostic() -> dict:
    """
    Etat de chargement du paquet.

    A appeler quand quelque chose renvoie None sans explication :

        python -c "import diarization; print(diarization.diagnostic())"

    Renvoie les sous-modules qui n'ont pas pu etre importes, et la
    liste des noms publics qui valent None faute d'avoir ete charges.
    """

    manquants = sorted(
        nom
        for nom in __all__
        if nom not in {"IMPORT_ERRORS", "diagnostic"}
        and globals().get(nom, False) is None
    )

    return {
        "version": __version__,
        "modules_en_echec": dict(IMPORT_ERRORS),
        "noms_a_None": manquants,
        "complet": not IMPORT_ERRORS and not manquants,
    }


if IMPORT_ERRORS:
    _warnings.warn(
        "diarization : %d sous-module(s) non charge(s) : %s. "
        "Detail : diarization.diagnostic()"
        % (
            len(IMPORT_ERRORS),
            ", ".join(sorted(IMPORT_ERRORS)),
        ),
        RuntimeWarning,
        stacklevel=2,
    )
