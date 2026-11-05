"""
====================================================================
SpeakerIdentity IA - Diarization Configuration
====================================================================

Projet       : SpeakerIdentity IA
Module       : Diarization
Fichier      : diarization/config.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 2.1.7
Date         : 14/08/2026

Description
------------

Configuration centrale du module de diarisation SpeakerIdentity IA.

Compatible avec :

    - session_manager.py
    - pipeline.py
    - audio_processor.py
    - vad.py
    - segment_manager.py
    - embedding_engine.py
    - speaker_manager.py

Format audio interne SpeakerIdentity :

    PCM16
    16000 Hz
    Mono
    2 bytes/sample

====================================================================
"""

from __future__ import annotations

import os

from dataclasses import dataclass


# ====================================================================
# PROJECT INFORMATION
# ====================================================================

PROJECT_NAME = "SpeakerIdentity IA"

MODULE_NAME = "Diarization"

MODULE_VERSION = "2.1.7"

AUTHOR = "SEBRI Abderrahim"

ORGANIZATION = "SAM TEHNOLOGY Lab"


# ====================================================================
# GENERAL
# ====================================================================

DEBUG = os.getenv(
    "SpeakerIdentity_DIARIZATION_DEBUG",
    "false",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


LOG_LEVEL = os.getenv(
    "SpeakerIdentity_DIARIZATION_LOG_LEVEL",
    "INFO",
).strip().upper()


# ====================================================================
# AUDIO FORMAT
# ====================================================================

SAMPLE_RATE = 16000

CHANNELS = 1

SAMPLE_WIDTH = 2

SAMPLE_FORMAT = "PCM16"

AUDIO_DTYPE = "<i2"

# CORRECTIF : __init__.py et les tests importaient AUDIO_CODEC et
# AUDIO_FORMAT, qui n'existaient nulle part. L'ImportError etait avalee
# par le try/except de __init__.py, ce qui mettait silencieusement a None
# tout le bloc config, y compris l'objet `config` lui-meme.
#
# AUDIO_CODEC  : le codec, tel qu'annonce par le navigateur.
# AUDIO_FORMAT : la description complete du format attendu par SpeakerIdentity.

AUDIO_CODEC = SAMPLE_FORMAT

AUDIO_FORMAT = {
    "codec": SAMPLE_FORMAT,
    "sample_rate": SAMPLE_RATE,
    "channels": CHANNELS,
    "sample_width": SAMPLE_WIDTH,
    "dtype": AUDIO_DTYPE,
}


# ====================================================================
# AUDIO BUFFER
# ====================================================================

AUDIO_BUFFER_MS = int(
    os.getenv(
        "SpeakerIdentity_AUDIO_BUFFER_MS",
        "100",
    )
)


AUDIO_BUFFER_MAX_SECONDS = float(
    os.getenv(
        "SpeakerIdentity_AUDIO_BUFFER_MAX_SECONDS",
        "10.0",
    )
)


# ====================================================================
# AUDIO PROCESSING
# ====================================================================

AUDIO_NORMALIZE = True

AUDIO_TARGET_SAMPLE_RATE = SAMPLE_RATE

AUDIO_TARGET_CHANNELS = CHANNELS

AUDIO_TARGET_SAMPLE_WIDTH = SAMPLE_WIDTH


# ====================================================================
# VAD
# ====================================================================

VAD_ENABLED = True


VAD_BACKEND = os.getenv(
    "SpeakerIdentity_VAD_BACKEND",
    "auto",
).strip().lower()


VAD_AGGRESSIVENESS = int(
    os.getenv(
        "SpeakerIdentity_VAD_AGGRESSIVENESS",
        "2",
    )
)


VAD_FRAME_DURATION_MS = int(
    os.getenv(
        "SpeakerIdentity_VAD_FRAME_DURATION_MS",
        "10",
    )
)


SILENCE_RMS = float(
    os.getenv(
        "SpeakerIdentity_SILENCE_RMS",
        "0.01",
    )
)


MIN_SPEECH_MS = int(
    os.getenv(
        "SpeakerIdentity_MIN_SPEECH_MS",
        "30",
    )
)


MIN_SILENCE_MS = int(
    os.getenv(
        "SpeakerIdentity_MIN_SILENCE_MS",
        "300",
    )
)


PADDING_MS = int(
    os.getenv(
        "SpeakerIdentity_PADDING_MS",
        "50",
    )
)


VAD_MIN_SPEECH_FRAMES = int(
    os.getenv(
        "SpeakerIdentity_VAD_MIN_SPEECH_FRAMES",
        "3",
    )
)


VAD_MIN_SILENCE_FRAMES = int(
    os.getenv(
        "SpeakerIdentity_VAD_MIN_SILENCE_FRAMES",
        "5",
    )
)


# ====================================================================
# VAD COMPATIBILITY
# ====================================================================

SEGMENT_PADDING_MS = PADDING_MS


# ====================================================================
# SEGMENTATION
# ====================================================================

MIN_SEGMENT_DURATION_MS = int(
    os.getenv(
        "SpeakerIdentity_MIN_SEGMENT_DURATION_MS",
        "300",
    )
)


MAX_SEGMENT_DURATION_MS = int(
    os.getenv(
        "SpeakerIdentity_MAX_SEGMENT_DURATION_MS",
        "10000",
    )
)


SEGMENT_SILENCE_DURATION_MS = int(
    os.getenv(
        "SpeakerIdentity_SEGMENT_SILENCE_DURATION_MS",
        "300",
    )
)


MAX_SEGMENTS_PER_SESSION = int(
    os.getenv(
        "SpeakerIdentity_MAX_SEGMENTS_PER_SESSION",
        "1000",
    )
)


# ====================================================================
# SEGMENT COMPATIBILITY
# ====================================================================

SEGMENT_MIN_DURATION_MS = MIN_SEGMENT_DURATION_MS

SEGMENT_MAX_DURATION_MS = MAX_SEGMENT_DURATION_MS


# ====================================================================
# EMBEDDING
# ====================================================================

EMBEDDING_ENABLED = True


EMBEDDING_MODEL = os.getenv(
    "SpeakerIdentity_EMBEDDING_MODEL",
    "speechbrain/spkrec-ecapa-voxceleb",
).strip()


EMBEDDING_DIMENSION = int(
    os.getenv(
        "SpeakerIdentity_EMBEDDING_DIMENSION",
        "192",
    )
)


EMBEDDING_SAMPLE_RATE = SAMPLE_RATE


EMBEDDING_MIN_AUDIO_MS = int(
    os.getenv(
        "SpeakerIdentity_EMBEDDING_MIN_AUDIO_MS",
        "500",
    )
)


EMBEDDING_MAX_AUDIO_MS = int(
    os.getenv(
        "SpeakerIdentity_EMBEDDING_MAX_AUDIO_MS",
        "3000",
    )
)


# ====================================================================
# COHERENCE SEGMENTATION / EMBEDDING
# ====================================================================
#
# CORRECTIF : MIN_SEGMENT_DURATION_MS valait 300 ms alors que
# EMBEDDING_MIN_AUDIO_MS en exige 500. Tout segment de 300 a 500 ms
# etait donc produit, transmis, puis rejete par le moteur
# d'embeddings avec "Audio is too short for speaker embedding" :
# du travail pour rien, et un segment de parole perdu.
#
# On aligne le plancher de segmentation sur ce que le moteur accepte
# reellement. Une valeur explicitement posee dans l'environnement
# reste prioritaire.

if "SpeakerIdentity_MIN_SEGMENT_DURATION_MS" not in os.environ:

    MIN_SEGMENT_DURATION_MS = max(
        MIN_SEGMENT_DURATION_MS,
        EMBEDDING_MIN_AUDIO_MS,
    )

SEGMENT_MIN_DURATION_MS = MIN_SEGMENT_DURATION_MS


# ====================================================================
# EMBEDDING INFERENCE
# ====================================================================

INFERENCE_BATCH_SIZE = int(
    os.getenv(
        "SpeakerIdentity_INFERENCE_BATCH_SIZE",
        "1",
    )
)


# ====================================================================
# DEVICE
# ====================================================================

DEVICE = os.getenv(
    "SpeakerIdentity_DIARIZATION_DEVICE",
    "cpu",
).strip().lower()


USE_GPU = os.getenv(
    "SpeakerIdentity_USE_GPU",
    "false",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


GPU_DEVICE_INDEX = int(
    os.getenv(
        "SpeakerIdentity_GPU_DEVICE_INDEX",
        "0",
    )
)


CPU_THREADS = int(
    os.getenv(
        "SpeakerIdentity_CPU_THREADS",
        str(
            max(
                1,
                min(
                    4,
                    os.cpu_count() or 1,
                ),
            )
        ),
    )
)


# ====================================================================
# SPEAKER IDENTIFICATION
# ====================================================================

SPEAKER_IDENTIFICATION_ENABLED = os.getenv(
    "SpeakerIdentity_SPEAKER_IDENTIFICATION_ENABLED",
    "true",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


SPEAKER_DISTANCE_THRESHOLD = float(
    os.getenv(
        "SpeakerIdentity_SPEAKER_DISTANCE_THRESHOLD",
        "0.35",
    )
)


SPEAKER_UNKNOWN_THRESHOLD = float(
    os.getenv(
        "SpeakerIdentity_SPEAKER_UNKNOWN_THRESHOLD",
        "0.60",
    )
)


SPEAKER_MIN_CONFIDENCE = float(
    os.getenv(
        "SpeakerIdentity_SPEAKER_MIN_CONFIDENCE",
        "0.50",
    )
)


SPEAKER_DISTANCE_METRIC = os.getenv(
    "SpeakerIdentity_SPEAKER_DISTANCE_METRIC",
    "cosine",
).strip().lower()


SPEAKER_UPDATE_ENABLED = os.getenv(
    "SpeakerIdentity_SPEAKER_UPDATE_ENABLED",
    "true",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


# CORRECTIF : ce seuil valait 0.65, ce qui est beaucoup trop haut pour
# la similarite cosinus brute d'ECAPA-TDNN.
#
# Constate a l'usage : un NOUVEAU locuteur etait cree a chaque prise de
# parole, y compris quand c'etait la meme personne qui parlait. Avec ce
# modele, deux extraits courts d'une meme voix tombent couramment entre
# 0,40 et 0,70 ; deux personnes differentes entre 0,00 et 0,30. Un seuil
# a 0,65 rejette donc une bonne partie des vraies correspondances.
#
# Reference : speechbrain compare ce meme modele avec un seuil de 0,25
# sur la similarite cosinus. 0,35 laisse une marge de securite pour la
# diarisation, ou les segments sont plus courts qu'en verification.
#
# CE CHIFFRE RESTE UN POINT DE DEPART. Il depend du micro, de la piece
# et des voix. Pour le regler sur des mesures reelles :
#
#     python calibrer_seuil.py <dossier de voix>
#
# puis poser la valeur trouvee :
#
#     set SpeakerIdentity_SPEAKER_SIMILARITY_THRESHOLD=0.42

SPEAKER_SIMILARITY_THRESHOLD = float(
    os.getenv(
        "SpeakerIdentity_SPEAKER_SIMILARITY_THRESHOLD",
        "0.35",
    )
)


# ====================================================================
# SPEAKER MEMORY
# ====================================================================

SPEAKER_MEMORY_ENABLED = True


SPEAKER_MEMORY_MAX_SPEAKERS = int(
    os.getenv(
        "SpeakerIdentity_SPEAKER_MEMORY_MAX_SPEAKERS",
        "100",
    )
)


SPEAKER_MEMORY_MAX_EMBEDDINGS_PER_SPEAKER = int(
    os.getenv(
        "SpeakerIdentity_SPEAKER_MEMORY_MAX_EMBEDDINGS_PER_SPEAKER",
        "20",
    )
)


SPEAKER_PROTOTYPE_SIZE = int(
    os.getenv(
        "SpeakerIdentity_SPEAKER_PROTOTYPE_SIZE",
        "5",
    )
)


# ====================================================================
# SESSION MANAGEMENT
# ====================================================================

SESSION_ENABLED = True


SESSION_MAX_ACTIVE = int(
    os.getenv(
        "SpeakerIdentity_SESSION_MAX_ACTIVE",
        "10",
    )
)


SESSION_TIMEOUT_SECONDS = int(
    os.getenv(
        "SpeakerIdentity_SESSION_TIMEOUT_SECONDS",
        "3600",
    )
)


# ====================================================================
# SESSION COMPATIBILITY
# ====================================================================

MAX_CONCURRENT_SESSIONS = SESSION_MAX_ACTIVE


MAX_SESSIONS = int(
    os.getenv(
        "SpeakerIdentity_MAX_SESSIONS",
        str(SESSION_MAX_ACTIVE),
    )
)


SESSION_MEMORY_TTL_SECONDS = int(
    os.getenv(
        "SpeakerIdentity_SESSION_MEMORY_TTL_SECONDS",
        str(SESSION_TIMEOUT_SECONDS),
    )
)


SESSION_LOCK_TIMEOUT_SECONDS = float(
    os.getenv(
        "SpeakerIdentity_SESSION_LOCK_TIMEOUT_SECONDS",
        "5.0",
    )
)


MULTI_SESSION_ENABLED = os.getenv(
    "SpeakerIdentity_MULTI_SESSION_ENABLED",
    "true",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


SESSION_ISOLATION_ENABLED = os.getenv(
    "SpeakerIdentity_SESSION_ISOLATION_ENABLED",
    "true",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


# ====================================================================
# SPEAKER SESSION LIMITS
# ====================================================================

MAX_SPEAKERS_PER_SESSION = int(
    os.getenv(
        "SpeakerIdentity_MAX_SPEAKERS_PER_SESSION",
        "50",
    )
)


# ====================================================================
# REAL-TIME PIPELINE
# ====================================================================

PIPELINE_ENABLED = True


PIPELINE_QUEUE_SIZE = int(
    os.getenv(
        "SpeakerIdentity_PIPELINE_QUEUE_SIZE",
        "100",
    )
)


PIPELINE_MAX_WORKERS = int(
    os.getenv(
        "SpeakerIdentity_PIPELINE_MAX_WORKERS",
        "1",
    )
)


# ====================================================================
# WEBSOCKET
# ====================================================================

WEBSOCKET_ENABLED = True


WEBSOCKET_HOST = os.getenv(
    "SpeakerIdentity_WEBSOCKET_HOST",
    "0.0.0.0",
).strip()


WEBSOCKET_PORT = int(
    os.getenv(
        "SpeakerIdentity_WEBSOCKET_PORT",
        "8010",
    )
)


WEBSOCKET_PATH = os.getenv(
    "SpeakerIdentity_WEBSOCKET_PATH",
    "/ws/diarization",
).strip()


# CORRECTIF : websocket.py lit config.websocket_max_message_bytes,
# api.py lit config.api_enabled, config.health_endpoint et
# config.stats_endpoint. Aucun de ces quatre attributs n'existait.
#
# Consequence : create_websocket_handler() et create_diarization_router()
# levaient AttributeError des la construction. La couche WebSocket et la
# couche REST etaient donc inutilisables, alors que ce sont precisement
# celles qu'utilise le serveur de demonstration.

WEBSOCKET_MAX_MESSAGE_BYTES = int(
    os.getenv(
        "SpeakerIdentity_WEBSOCKET_MAX_MESSAGE_BYTES",
        # 1 Mo : tres au-dessus d'un bloc PCM16 de 100 ms (3200 octets),
        # assez bas pour rejeter un message aberrant.
        str(1024 * 1024),
    )
)


API_ENABLED = os.getenv(
    "SpeakerIdentity_DIARIZATION_API_ENABLED",
    "true",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


STATS_ENDPOINT = os.getenv(
    "SpeakerIdentity_DIARIZATION_STATS_ENDPOINT",
    "/statistics",
)


# ====================================================================
# API
# ====================================================================

API_HOST = os.getenv(
    "SpeakerIdentity_DIARIZATION_API_HOST",
    "0.0.0.0",
).strip()


API_PORT = int(
    os.getenv(
        "SpeakerIdentity_DIARIZATION_API_PORT",
        "8011",
    )
)


API_PREFIX = os.getenv(
    "SpeakerIdentity_DIARIZATION_API_PREFIX",
    "/api",
).strip()


# ====================================================================
# HEALTH
# ====================================================================

HEALTH_ENABLED = True

HEALTH_PATH = os.getenv(
    "SpeakerIdentity_DIARIZATION_HEALTH_PATH",
    "/health",
).strip()


# ====================================================================
# METRICS
# ====================================================================

METRICS_ENABLED = True


# ====================================================================
# DIARIZATION CONFIGURATION OBJECT
# ====================================================================

@dataclass
class DiarizationConfig:
    """
    Objet de configuration utilisé par les composants
    du pipeline de diarisation SpeakerIdentity IA.
    """

    # ---------------------------------------------------------------
    # General
    # ---------------------------------------------------------------

    debug: bool = DEBUG

    log_level: str = LOG_LEVEL

    # ---------------------------------------------------------------
    # Audio
    # ---------------------------------------------------------------

    sample_rate: int = SAMPLE_RATE

    channels: int = CHANNELS

    sample_width: int = SAMPLE_WIDTH

    sample_format: str = SAMPLE_FORMAT

    audio_dtype: str = AUDIO_DTYPE

    audio_normalize: bool = AUDIO_NORMALIZE

    audio_buffer_ms: int = AUDIO_BUFFER_MS

    audio_buffer_max_seconds: float = (
        AUDIO_BUFFER_MAX_SECONDS
    )

    audio_target_sample_rate: int = (
        AUDIO_TARGET_SAMPLE_RATE
    )

    audio_target_channels: int = (
        AUDIO_TARGET_CHANNELS
    )

    audio_target_sample_width: int = (
        AUDIO_TARGET_SAMPLE_WIDTH
    )

    # ---------------------------------------------------------------
    # VAD
    # ---------------------------------------------------------------

    vad_enabled: bool = VAD_ENABLED

    vad_backend: str = VAD_BACKEND

    vad_aggressiveness: int = (
        VAD_AGGRESSIVENESS
    )

    frame_duration_ms: int = (
        VAD_FRAME_DURATION_MS
    )

    # Compatibility name
    vad_frame_duration_ms: int = (
        VAD_FRAME_DURATION_MS
    )

    silence_rms: float = SILENCE_RMS

    min_speech_ms: int = MIN_SPEECH_MS

    min_silence_ms: int = MIN_SILENCE_MS

    padding_ms: int = PADDING_MS

    # Compatibility name
    segment_padding_ms: int = (
        SEGMENT_PADDING_MS
    )

    vad_min_speech_frames: int = (
        VAD_MIN_SPEECH_FRAMES
    )

    vad_min_silence_frames: int = (
        VAD_MIN_SILENCE_FRAMES
    )

    # ---------------------------------------------------------------
    # Segmentation
    # ---------------------------------------------------------------

    min_segment_duration_ms: int = (
        MIN_SEGMENT_DURATION_MS
    )

    max_segment_duration_ms: int = (
        MAX_SEGMENT_DURATION_MS
    )

    # Compatibility names
    segment_min_duration_ms: int = (
        SEGMENT_MIN_DURATION_MS
    )

    segment_max_duration_ms: int = (
        SEGMENT_MAX_DURATION_MS
    )

    segment_silence_duration_ms: int = (
        SEGMENT_SILENCE_DURATION_MS
    )

    max_segments_per_session: int = (
        MAX_SEGMENTS_PER_SESSION
    )

    # ---------------------------------------------------------------
    # Embedding
    # ---------------------------------------------------------------

    embedding_enabled: bool = (
        EMBEDDING_ENABLED
    )

    embedding_model: str = EMBEDDING_MODEL

    embedding_dimension: int = (
        EMBEDDING_DIMENSION
    )

    embedding_sample_rate: int = (
        EMBEDDING_SAMPLE_RATE
    )

    embedding_min_audio_ms: int = (
        EMBEDDING_MIN_AUDIO_MS
    )

    embedding_max_audio_ms: int = (
        EMBEDDING_MAX_AUDIO_MS
    )

    inference_batch_size: int = (
        INFERENCE_BATCH_SIZE
    )

    # ---------------------------------------------------------------
    # Device
    # ---------------------------------------------------------------

    device: str = DEVICE

    use_gpu: bool = USE_GPU

    gpu_device_index: int = (
        GPU_DEVICE_INDEX
    )

    cpu_threads: int = CPU_THREADS

    # ---------------------------------------------------------------
    # Speaker identification
    # ---------------------------------------------------------------

    speaker_identification_enabled: bool = (
        SPEAKER_IDENTIFICATION_ENABLED
    )

    speaker_distance_threshold: float = (
        SPEAKER_DISTANCE_THRESHOLD
    )

    speaker_unknown_threshold: float = (
        SPEAKER_UNKNOWN_THRESHOLD
    )

    speaker_min_confidence: float = (
        SPEAKER_MIN_CONFIDENCE
    )

    speaker_similarity_threshold: float = (
        SPEAKER_SIMILARITY_THRESHOLD
    )

    speaker_distance_metric: str = (
        SPEAKER_DISTANCE_METRIC
    )

    speaker_update_enabled: bool = (
        SPEAKER_UPDATE_ENABLED
    )

    # ---------------------------------------------------------------
    # Speaker memory
    # ---------------------------------------------------------------

    speaker_memory_enabled: bool = (
        SPEAKER_MEMORY_ENABLED
    )

    speaker_memory_max_speakers: int = (
        SPEAKER_MEMORY_MAX_SPEAKERS
    )

    speaker_memory_max_embeddings_per_speaker: int = (
        SPEAKER_MEMORY_MAX_EMBEDDINGS_PER_SPEAKER
    )

    speaker_prototype_size: int = (
        SPEAKER_PROTOTYPE_SIZE
    )

    # ---------------------------------------------------------------
    # Session
    # ---------------------------------------------------------------

    session_enabled: bool = SESSION_ENABLED

    session_max_active: int = (
        SESSION_MAX_ACTIVE
    )

    max_sessions: int = MAX_SESSIONS

    max_concurrent_sessions: int = (
        MAX_CONCURRENT_SESSIONS
    )

    session_timeout_seconds: int = (
        SESSION_TIMEOUT_SECONDS
    )

    session_memory_ttl_seconds: int = (
        SESSION_MEMORY_TTL_SECONDS
    )

    session_lock_timeout_seconds: float = (
        SESSION_LOCK_TIMEOUT_SECONDS
    )

    multi_session_enabled: bool = (
        MULTI_SESSION_ENABLED
    )

    session_isolation_enabled: bool = (
        SESSION_ISOLATION_ENABLED
    )

    max_speakers_per_session: int = (
        MAX_SPEAKERS_PER_SESSION
    )

    # ---------------------------------------------------------------
    # Pipeline
    # ---------------------------------------------------------------

    pipeline_enabled: bool = (
        PIPELINE_ENABLED
    )

    pipeline_queue_size: int = (
        PIPELINE_QUEUE_SIZE
    )

    pipeline_max_workers: int = (
        PIPELINE_MAX_WORKERS
    )

    # ---------------------------------------------------------------
    # WebSocket
    # ---------------------------------------------------------------

    websocket_enabled: bool = (
        WEBSOCKET_ENABLED
    )

    websocket_host: str = (
        WEBSOCKET_HOST
    )

    websocket_port: int = (
        WEBSOCKET_PORT
    )

    websocket_path: str = (
        WEBSOCKET_PATH
    )

    websocket_max_message_bytes: int = (
        WEBSOCKET_MAX_MESSAGE_BYTES
    )

    # ---------------------------------------------------------------
    # API
    # ---------------------------------------------------------------

    api_host: str = API_HOST

    api_port: int = API_PORT

    api_prefix: str = API_PREFIX

    api_enabled: bool = API_ENABLED

    stats_endpoint: str = STATS_ENDPOINT

    # ---------------------------------------------------------------
    # Health
    # ---------------------------------------------------------------

    health_enabled: bool = (
        HEALTH_ENABLED
    )

    health_path: str = HEALTH_PATH

    # api.py utilise le nom health_endpoint ; on expose les deux.
    health_endpoint: str = HEALTH_PATH

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    metrics_enabled: bool = (
        METRICS_ENABLED
    )

    # =================================================================
    # VALIDATE
    # =================================================================

    def validate(self) -> None:
        """
        Valide la configuration utilisée par le pipeline.
        """

        # -------------------------------------------------------------
        # General
        # -------------------------------------------------------------

        if not self.log_level:
            raise ValueError(
                "log_level cannot be empty."
            )

        # -------------------------------------------------------------
        # Audio
        # -------------------------------------------------------------

        if self.sample_rate <= 0:
            raise ValueError(
                "sample_rate must be greater than zero."
            )

        if self.channels != 1:
            raise ValueError(
                "Diarization requires mono audio."
            )

        if self.sample_width != 2:
            raise ValueError(
                "Diarization requires PCM16 audio."
            )

        if self.sample_format.upper() != "PCM16":
            raise ValueError(
                "sample_format must be PCM16."
            )

        if self.audio_buffer_ms <= 0:
            raise ValueError(
                "audio_buffer_ms must be greater than zero."
            )

        if self.audio_buffer_max_seconds <= 0:
            raise ValueError(
                "audio_buffer_max_seconds must be greater than zero."
            )

        if (
            self.audio_buffer_ms / 1000
            > self.audio_buffer_max_seconds
        ):
            raise ValueError(
                "audio_buffer_ms cannot exceed "
                "audio_buffer_max_seconds."
            )

        if self.audio_target_sample_rate <= 0:
            raise ValueError(
                "audio_target_sample_rate must be greater than zero."
            )

        if self.audio_target_channels != 1:
            raise ValueError(
                "audio_target_channels must be mono."
            )

        if self.audio_target_sample_width != 2:
            raise ValueError(
                "audio_target_sample_width must be PCM16."
            )

        # -------------------------------------------------------------
        # VAD
        # -------------------------------------------------------------

        if self.vad_backend not in {
            "webrtc",
            "rms",
            "auto",
        }:
            raise ValueError(
                "vad_backend must be "
                "webrtc, rms or auto."
            )

        if self.vad_aggressiveness not in {
            0,
            1,
            2,
            3,
        }:
            raise ValueError(
                "vad_aggressiveness must be "
                "between 0 and 3."
            )

        if self.frame_duration_ms not in {
            10,
            20,
            30,
        }:
            raise ValueError(
                "frame_duration_ms must be "
                "10, 20 or 30."
            )

        if self.vad_frame_duration_ms not in {
            10,
            20,
            30,
        }:
            raise ValueError(
                "vad_frame_duration_ms must be "
                "10, 20 or 30."
            )

        if (
            self.frame_duration_ms
            != self.vad_frame_duration_ms
        ):
            raise ValueError(
                "frame_duration_ms and "
                "vad_frame_duration_ms must be identical."
            )

        if self.min_speech_ms < 0:
            raise ValueError(
                "min_speech_ms must be >= 0."
            )

        if self.min_silence_ms < 0:
            raise ValueError(
                "min_silence_ms must be >= 0."
            )

        if self.padding_ms < 0:
            raise ValueError(
                "padding_ms must be >= 0."
            )

        if self.segment_padding_ms < 0:
            raise ValueError(
                "segment_padding_ms must be >= 0."
            )

        if (
            self.padding_ms
            != self.segment_padding_ms
        ):
            raise ValueError(
                "padding_ms and segment_padding_ms "
                "must be identical."
            )

        if self.silence_rms < 0:
            raise ValueError(
                "silence_rms must be >= 0."
            )

        if self.vad_min_speech_frames < 0:
            raise ValueError(
                "vad_min_speech_frames must be >= 0."
            )

        if self.vad_min_silence_frames < 0:
            raise ValueError(
                "vad_min_silence_frames must be >= 0."
            )

        # -------------------------------------------------------------
        # Segmentation
        # -------------------------------------------------------------

        if self.min_segment_duration_ms <= 0:
            raise ValueError(
                "min_segment_duration_ms must "
                "be greater than zero."
            )

        if self.max_segment_duration_ms <= 0:
            raise ValueError(
                "max_segment_duration_ms must "
                "be greater than zero."
            )

        if (
            self.max_segment_duration_ms
            < self.min_segment_duration_ms
        ):
            raise ValueError(
                "max_segment_duration_ms must be "
                ">= min_segment_duration_ms."
            )

        if self.segment_min_duration_ms <= 0:
            raise ValueError(
                "segment_min_duration_ms must "
                "be greater than zero."
            )

        if self.segment_max_duration_ms <= 0:
            raise ValueError(
                "segment_max_duration_ms must "
                "be greater than zero."
            )

        if (
            self.segment_max_duration_ms
            < self.segment_min_duration_ms
        ):
            raise ValueError(
                "segment_max_duration_ms must be "
                ">= segment_min_duration_ms."
            )

        if (
            self.min_segment_duration_ms
            != self.segment_min_duration_ms
        ):
            raise ValueError(
                "min_segment_duration_ms and "
                "segment_min_duration_ms must be identical."
            )

        if (
            self.max_segment_duration_ms
            != self.segment_max_duration_ms
        ):
            raise ValueError(
                "max_segment_duration_ms and "
                "segment_max_duration_ms must be identical."
            )

        if self.segment_silence_duration_ms < 0:
            raise ValueError(
                "segment_silence_duration_ms must be >= 0."
            )

        if self.max_segments_per_session <= 0:
            raise ValueError(
                "max_segments_per_session must "
                "be greater than zero."
            )

        # -------------------------------------------------------------
        # Embedding
        # -------------------------------------------------------------

        if self.embedding_enabled:

            if not self.embedding_model:
                raise ValueError(
                    "embedding_model cannot be empty."
                )

            if self.embedding_dimension <= 0:
                raise ValueError(
                    "embedding_dimension must "
                    "be greater than zero."
                )

            if self.embedding_sample_rate <= 0:
                raise ValueError(
                    "embedding_sample_rate must "
                    "be greater than zero."
                )

            if self.embedding_min_audio_ms <= 0:
                raise ValueError(
                    "embedding_min_audio_ms must "
                    "be greater than zero."
                )

            if self.embedding_max_audio_ms <= 0:
                raise ValueError(
                    "embedding_max_audio_ms must "
                    "be greater than zero."
                )

            if (
                self.embedding_max_audio_ms
                < self.embedding_min_audio_ms
            ):
                raise ValueError(
                    "embedding_max_audio_ms must be "
                    ">= embedding_min_audio_ms."
                )

        if self.inference_batch_size <= 0:
            raise ValueError(
                "inference_batch_size must "
                "be greater than zero."
            )

        # -------------------------------------------------------------
        # Device
        # -------------------------------------------------------------

        if self.device not in {
            "cpu",
            "cuda",
            "auto",
        }:
            raise ValueError(
                "device must be cpu, cuda or auto."
            )

        if self.gpu_device_index < 0:
            raise ValueError(
                "gpu_device_index cannot be negative."
            )

        if self.cpu_threads <= 0:
            raise ValueError(
                "cpu_threads must be greater than zero."
            )

        # GPU explicit activation requires a CUDA device.
        if self.use_gpu and self.device == "cpu":
            raise ValueError(
                "use_gpu cannot be enabled when "
                "device is explicitly set to 'cpu'."
            )

        # -------------------------------------------------------------
        # Speaker identification
        # -------------------------------------------------------------

        if self.speaker_distance_threshold < 0:
            raise ValueError(
                "speaker_distance_threshold "
                "must be >= 0."
            )

        if self.speaker_unknown_threshold < 0:
            raise ValueError(
                "speaker_unknown_threshold "
                "must be >= 0."
            )

        if (
            self.speaker_unknown_threshold
            < self.speaker_distance_threshold
        ):
            raise ValueError(
                "speaker_unknown_threshold must "
                "be >= speaker_distance_threshold."
            )

        if not (
            0.0
            <= self.speaker_min_confidence
            <= 1.0
        ):
            raise ValueError(
                "speaker_min_confidence must "
                "be between 0 and 1."
            )

        if not (
            0.0
            <= self.speaker_similarity_threshold
            <= 1.0
        ):
            raise ValueError(
                "speaker_similarity_threshold must "
                "be between 0 and 1."
            )

        if self.speaker_distance_metric not in {
            "cosine",
            "euclidean",
        }:
            raise ValueError(
                "speaker_distance_metric must be "
                "'cosine' or 'euclidean'."
            )

        # -------------------------------------------------------------
        # Speaker memory
        # -------------------------------------------------------------

        if self.speaker_memory_max_speakers <= 0:
            raise ValueError(
                "speaker_memory_max_speakers must "
                "be greater than zero."
            )

        if (
            self.speaker_memory_max_embeddings_per_speaker
            <= 0
        ):
            raise ValueError(
                "speaker_memory_max_embeddings_per_speaker "
                "must be greater than zero."
            )

        if self.speaker_prototype_size <= 0:
            raise ValueError(
                "speaker_prototype_size must "
                "be greater than zero."
            )

        if (
            self.speaker_prototype_size
            > self.speaker_memory_max_embeddings_per_speaker
        ):
            raise ValueError(
                "speaker_prototype_size must be "
                "<= speaker_memory_max_embeddings_per_speaker."
            )

        # -------------------------------------------------------------
        # Session
        # -------------------------------------------------------------

        if self.session_max_active <= 0:
            raise ValueError(
                "session_max_active must be greater than zero."
            )

        if self.max_sessions <= 0:
            raise ValueError(
                "max_sessions must be greater than zero."
            )

        if self.max_concurrent_sessions <= 0:
            raise ValueError(
                "max_concurrent_sessions must "
                "be greater than zero."
            )

        if (
            self.max_concurrent_sessions
            != self.session_max_active
        ):
            raise ValueError(
                "max_concurrent_sessions and "
                "session_max_active must be identical."
            )

        if self.session_timeout_seconds < 0:
            raise ValueError(
                "session_timeout_seconds must be >= 0."
            )

        if self.session_memory_ttl_seconds < 0:
            raise ValueError(
                "session_memory_ttl_seconds "
                "must be >= 0."
            )

        if self.session_lock_timeout_seconds <= 0:
            raise ValueError(
                "session_lock_timeout_seconds "
                "must be greater than zero."
            )

        if self.max_speakers_per_session <= 0:
            raise ValueError(
                "max_speakers_per_session must "
                "be greater than zero."
            )

        # -------------------------------------------------------------
        # Pipeline
        # -------------------------------------------------------------

        if self.pipeline_queue_size <= 0:
            raise ValueError(
                "pipeline_queue_size must "
                "be greater than zero."
            )

        if self.pipeline_max_workers <= 0:
            raise ValueError(
                "pipeline_max_workers must "
                "be greater than zero."
            )

        # -------------------------------------------------------------
        # WebSocket
        # -------------------------------------------------------------

        if not self.websocket_host:
            raise ValueError(
                "websocket_host cannot be empty."
            )

        if not 1 <= self.websocket_port <= 65535:
            raise ValueError(
                "websocket_port must be between 1 and 65535."
            )

        if not self.websocket_path:
            raise ValueError(
                "websocket_path cannot be empty."
            )

        if not self.websocket_path.startswith("/"):
            raise ValueError(
                "websocket_path must start with '/'."
            )

        # -------------------------------------------------------------
        # API
        # -------------------------------------------------------------

        if not self.api_host:
            raise ValueError(
                "api_host cannot be empty."
            )

        if not 1 <= self.api_port <= 65535:
            raise ValueError(
                "api_port must be between 1 and 65535."
            )

        if not self.api_prefix:
            raise ValueError(
                "api_prefix cannot be empty."
            )

        if not self.api_prefix.startswith("/"):
            raise ValueError(
                "api_prefix must start with '/'."
            )

        # Prevent accidental identical network endpoints.
        if (
            self.websocket_enabled
            and self.websocket_host == self.api_host
            and self.websocket_port == self.api_port
        ):
            raise ValueError(
                "WebSocket and API cannot use the same "
                "host and port."
            )

        # -------------------------------------------------------------
        # Health
        # -------------------------------------------------------------

        if self.health_enabled:

            if not self.health_path:
                raise ValueError(
                    "health_path cannot be empty."
                )

            if not self.health_path.startswith("/"):
                raise ValueError(
                    "health_path must start with '/'."
                )

        # -------------------------------------------------------------
        # Metrics
        # -------------------------------------------------------------

        if not isinstance(
            self.metrics_enabled,
            bool,
        ):
            raise ValueError(
                "metrics_enabled must be a boolean."
            )


# ====================================================================
# DEFAULT CONFIGURATION INSTANCE
# ====================================================================

config = DiarizationConfig()

# CORRECTIF : __init__.py importait `diarization_config`, qui n'existait
# pas. Meme cause, meme effet que ci-dessus : tout le bloc config passait
# a None. On expose l'alias explicitement plutot que de retirer le nom,
# car il est plus parlant a l'usage depuis un autre module.

diarization_config = config


# ====================================================================
# GLOBAL VALIDATION
# ====================================================================

config.validate()


# ====================================================================
# PUBLIC EXPORTS
# ====================================================================

__all__ = [

    # ---------------------------------------------------------------
    # Project
    # ---------------------------------------------------------------

    "PROJECT_NAME",
    "MODULE_NAME",
    "MODULE_VERSION",
    "AUTHOR",
    "ORGANIZATION",

    # ---------------------------------------------------------------
    # General
    # ---------------------------------------------------------------

    "DEBUG",
    "LOG_LEVEL",

    # ---------------------------------------------------------------
    # Audio
    # ---------------------------------------------------------------

    "SAMPLE_RATE",
    "CHANNELS",
    "SAMPLE_WIDTH",
    "SAMPLE_FORMAT",
    "AUDIO_DTYPE",

    # ---------------------------------------------------------------
    # Buffer
    # ---------------------------------------------------------------

    "AUDIO_BUFFER_MS",
    "AUDIO_BUFFER_MAX_SECONDS",

    # ---------------------------------------------------------------
    # Processing
    # ---------------------------------------------------------------

    "AUDIO_NORMALIZE",
    "AUDIO_TARGET_SAMPLE_RATE",
    "AUDIO_TARGET_CHANNELS",
    "AUDIO_TARGET_SAMPLE_WIDTH",

    # ---------------------------------------------------------------
    # VAD
    # ---------------------------------------------------------------

    "VAD_ENABLED",
    "VAD_BACKEND",
    "VAD_AGGRESSIVENESS",
    "VAD_FRAME_DURATION_MS",
    "VAD_MIN_SPEECH_FRAMES",
    "VAD_MIN_SILENCE_FRAMES",
    "SILENCE_RMS",
    "MIN_SPEECH_MS",
    "MIN_SILENCE_MS",
    "PADDING_MS",
    "SEGMENT_PADDING_MS",

    # ---------------------------------------------------------------
    # Segmentation
    # ---------------------------------------------------------------

    "MIN_SEGMENT_DURATION_MS",
    "MAX_SEGMENT_DURATION_MS",
    "SEGMENT_SILENCE_DURATION_MS",
    "MAX_SEGMENTS_PER_SESSION",
    "SEGMENT_MIN_DURATION_MS",
    "SEGMENT_MAX_DURATION_MS",

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

    "SPEAKER_IDENTIFICATION_ENABLED",
    "SPEAKER_DISTANCE_THRESHOLD",
    "SPEAKER_UNKNOWN_THRESHOLD",
    "SPEAKER_MIN_CONFIDENCE",
    "SPEAKER_SIMILARITY_THRESHOLD",
    "SPEAKER_DISTANCE_METRIC",
    "SPEAKER_UPDATE_ENABLED",

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
    "MAX_SESSIONS",
    "MAX_CONCURRENT_SESSIONS",
    "SESSION_TIMEOUT_SECONDS",
    "SESSION_MEMORY_TTL_SECONDS",
    "SESSION_LOCK_TIMEOUT_SECONDS",
    "MULTI_SESSION_ENABLED",
    "SESSION_ISOLATION_ENABLED",
    "MAX_SPEAKERS_PER_SESSION",

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
    "WEBSOCKET_MAX_MESSAGE_BYTES",

    # ---------------------------------------------------------------
    # API
    # ---------------------------------------------------------------

    "API_HOST",
    "API_PORT",
    "API_PREFIX",
    "API_ENABLED",
    "STATS_ENDPOINT",

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
    # Configuration object
    # ---------------------------------------------------------------

    "DiarizationConfig",
    "config",
    "diarization_config",

    # ---------------------------------------------------------------
    # Format audio
    # ---------------------------------------------------------------

    "AUDIO_CODEC",
    "AUDIO_FORMAT",
]