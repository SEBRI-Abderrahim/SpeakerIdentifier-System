"""
====================================================================
SpeakerIdentity IA - Diarization Exceptions
====================================================================

Projet       : SpeakerIdentity IA
Module       : Diarization
Fichier      : diarization/exceptions.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 2.2.0
Date         : 14/08/2026

Description
------------

Exceptions communes au module de diarisation SpeakerIdentity IA.

Cette hiérarchie fournit un contrat d'erreur commun pour :

    - audio_buffer.py
    - audio_processor.py
    - vad.py
    - segment_manager.py
    - embedding_engine.py
    - speaker_memory.py
    - speaker_manager.py
    - session_manager.py
    - pipeline.py
    - diarization_engine.py
    - websocket.py
    - api.py
    - health.py
    - metrics.py

Objectifs :

    - fournir une hiérarchie d'erreurs cohérente ;
    - faciliter le diagnostic ;
    - conserver un code d'erreur stable ;
    - transporter des détails structurés ;
    - permettre la sérialisation REST/WebSocket ;
    - distinguer les erreurs fonctionnelles ;
    - préserver l'exception originale lors des encapsulations.

Architecture :

    DiarizationError
        |
        +-- ConfigurationError
        |
        +-- AudioError
        |     +-- AudioFormatError
        |     +-- AudioBufferError
        |     +-- AudioProcessingError
        |
        +-- VADError
        |     +-- VADConfigurationError
        |     +-- VADProcessingError
        |
        +-- SegmentationError
        |     +-- InvalidSegmentError
        |     +-- SegmentLimitError
        |
        +-- EmbeddingError
        |     +-- EmbeddingModelError
        |     +-- EmbeddingDimensionError
        |
        +-- SpeakerError
        |     +-- SpeakerIdentificationError
        |     +-- SpeakerMemoryError
        |     +-- SpeakerLimitError
        |
        +-- SessionError
        |     +-- SessionNotFoundError
        |     +-- SessionAlreadyExistsError
        |     +-- SessionLimitError
        |     +-- SessionStateError
        |     +-- SessionIsolationError
        |
        +-- PipelineError
        |     +-- PipelineStateError
        |     +-- PipelineTimeoutError
        |
        +-- WebSocketError
        |     +-- WebSocketProtocolError
        |     +-- WebSocketPayloadError
        |
        +-- APIError
              +-- APIValidationError

====================================================================
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
)


# ====================================================================
# ROOT EXCEPTION
# ====================================================================


class DiarizationError(Exception):
    """
    Exception racine du module SpeakerIdentity Diarization.

    Toutes les exceptions fonctionnelles du module doivent dériver
    de cette classe.

    Parameters
    ----------
    message:
        Message humainement lisible.

    code:
        Code d'erreur personnalisé. Si absent, le code défini par
        la classe est utilisé.

    details:
        Informations structurées supplémentaires.
    """

    error_code = "DIARIZATION_ERROR"

    def __init__(
        self,
        message: str = "",
        *,
        code: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        # ------------------------------------------------------------
        # MESSAGE
        # ------------------------------------------------------------

        self.message = (
            str(message).strip()
            if message is not None
            else ""
        )

        if not self.message:
            self.message = self.__class__.__name__

        # ------------------------------------------------------------
        # ERROR CODE
        # ------------------------------------------------------------

        default_code = str(
            getattr(
                self.__class__,
                "error_code",
                DiarizationError.error_code,
            )
        ).strip()

        if not default_code:
            default_code = DiarizationError.error_code

        if code is None:
            normalized_code = default_code

        else:
            normalized_code = str(
                code
            ).strip()

            if not normalized_code:
                normalized_code = default_code

        self.code = normalized_code

        # ------------------------------------------------------------
        # DETAILS
        # ------------------------------------------------------------

        if details is None:
            self.details: Dict[str, Any] = {}

        elif isinstance(
            details,
            Mapping,
        ):
            self.details = dict(details)

        else:
            raise TypeError(
                "details must be a mapping or None."
            )

        # ------------------------------------------------------------
        # BASE EXCEPTION
        # ------------------------------------------------------------

        super().__init__(
            self.message
        )

    # =================================================================
    # SERIALIZATION
    # =================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit l'exception en structure sérialisable.

        Compatible avec :

            - REST API
            - WebSocket
            - logs
            - monitoring
            - tests
        """

        return {
            "error": self.code,
            "message": self.message,
            "details": dict(
                self.details
            ),
        }

    # =================================================================
    # STRING REPRESENTATION
    # =================================================================

    def __str__(self) -> str:
        """
        Retourne le message utilisateur.
        """

        return self.message

    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Représentation développeur.
        """

        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"code={self.code!r}, "
            f"details={self.details!r}"
            f")"
        )


# ====================================================================
# CONFIGURATION
# ====================================================================


class ConfigurationError(DiarizationError):
    """
    Erreur de configuration du module.
    """

    error_code = (
        "DIARIZATION_CONFIGURATION_ERROR"
    )


# ====================================================================
# AUDIO ERRORS
# ====================================================================


class AudioError(DiarizationError):
    """
    Exception racine pour les erreurs audio.
    """

    error_code = (
        "DIARIZATION_AUDIO_ERROR"
    )


class AudioFormatError(AudioError):
    """
    Format audio incompatible avec SpeakerIdentity.

    Exemples :

        - mauvais sample rate ;
        - mauvais nombre de canaux ;
        - mauvais sample width ;
        - codec non supporté.
    """

    error_code = "AUDIO_FORMAT_ERROR"


class AudioBufferError(AudioError):
    """
    Erreur liée au buffer audio.
    """

    error_code = "AUDIO_BUFFER_ERROR"


class AudioProcessingError(AudioError):
    """
    Erreur pendant le traitement ou la conversion audio.
    """

    error_code = "AUDIO_PROCESSING_ERROR"


# ====================================================================
# VAD
# ====================================================================


class VADError(DiarizationError):
    """
    Erreur liée au système de Voice Activity Detection.
    """

    error_code = "VAD_ERROR"


class VADConfigurationError(VADError):
    """
    Configuration VAD invalide.
    """

    error_code = "VAD_CONFIGURATION_ERROR"


class VADProcessingError(VADError):
    """
    Erreur pendant le traitement d'une frame audio.
    """

    error_code = "VAD_PROCESSING_ERROR"


# ====================================================================
# SEGMENTATION
# ====================================================================


class SegmentationError(DiarizationError):
    """
    Erreur liée à la segmentation audio.
    """

    error_code = "SEGMENTATION_ERROR"


class InvalidSegmentError(SegmentationError):
    """
    Segment audio invalide.

    Exemples :

        - end < start ;
        - durée négative ;
        - timestamps incohérents.
    """

    error_code = "INVALID_SEGMENT_ERROR"


class SegmentLimitError(SegmentationError):
    """
    Limite de segments atteinte pour une session.
    """

    error_code = "SEGMENT_LIMIT_ERROR"


# ====================================================================
# EMBEDDING
# ====================================================================


class EmbeddingError(DiarizationError):
    """
    Erreur de génération ou de manipulation d'embedding vocal.
    """

    error_code = "EMBEDDING_ERROR"


class EmbeddingModelError(EmbeddingError):
    """
    Erreur de chargement ou d'exécution du modèle d'embedding.
    """

    error_code = "EMBEDDING_MODEL_ERROR"


class EmbeddingDimensionError(EmbeddingError):
    """
    Dimension d'embedding incompatible.
    """

    error_code = "EMBEDDING_DIMENSION_ERROR"


# ====================================================================
# SPEAKER
# ====================================================================


class SpeakerError(DiarizationError):
    """
    Exception racine liée aux speakers.
    """

    error_code = "SPEAKER_ERROR"


class SpeakerIdentificationError(SpeakerError):
    """
    Erreur d'identification d'un speaker.
    """

    error_code = (
        "SPEAKER_IDENTIFICATION_ERROR"
    )


class SpeakerMemoryError(SpeakerError):
    """
    Erreur de mémoire vocale.
    """

    error_code = "SPEAKER_MEMORY_ERROR"


class SpeakerLimitError(SpeakerError):
    """
    Limite de speakers atteinte dans une session.
    """

    error_code = "SPEAKER_LIMIT_ERROR"


# ====================================================================
# SESSION
# ====================================================================


class SessionError(DiarizationError):
    """
    Exception racine liée aux sessions.
    """

    error_code = "SESSION_ERROR"


class SessionNotFoundError(SessionError):
    """
    Session inexistante.
    """

    error_code = "SESSION_NOT_FOUND"


class SessionAlreadyExistsError(SessionError):
    """
    Tentative de création d'une session existante.
    """

    error_code = "SESSION_ALREADY_EXISTS"


class SessionLimitError(SessionError):
    """
    Nombre maximal de sessions atteint.
    """

    error_code = "SESSION_LIMIT_REACHED"


class SessionStateError(SessionError):
    """
    Opération incompatible avec l'état courant de la session.
    """

    error_code = "SESSION_STATE_ERROR"


class SessionIsolationError(SessionError):
    """
    Violation de l'isolation entre sessions.
    """

    error_code = "SESSION_ISOLATION_ERROR"


# ====================================================================
# PIPELINE
# ====================================================================


class PipelineError(DiarizationError):
    """
    Erreur générale du pipeline.
    """

    error_code = "PIPELINE_ERROR"


class PipelineStateError(PipelineError):
    """
    État du pipeline incompatible avec l'opération demandée.
    """

    error_code = "PIPELINE_STATE_ERROR"


class PipelineTimeoutError(PipelineError):
    """
    Timeout pendant le traitement.
    """

    error_code = "PIPELINE_TIMEOUT"


# ====================================================================
# WEBSOCKET
# ====================================================================


class WebSocketError(DiarizationError):
    """
    Erreur liée au transport WebSocket.
    """

    error_code = "WEBSOCKET_ERROR"


class WebSocketProtocolError(WebSocketError):
    """
    Message WebSocket invalide.
    """

    error_code = "WEBSOCKET_PROTOCOL_ERROR"


class WebSocketPayloadError(WebSocketError):
    """
    Payload WebSocket invalide ou trop volumineux.
    """

    error_code = "WEBSOCKET_PAYLOAD_ERROR"


# ====================================================================
# API
# ====================================================================


class APIError(DiarizationError):
    """
    Erreur exposée par la couche API.
    """

    error_code = "API_ERROR"


class APIValidationError(APIError):
    """
    Paramètres API invalides.
    """

    error_code = "API_VALIDATION_ERROR"


# ====================================================================
# UTILITY
# ====================================================================


def exception_to_dict(
    exc: Exception,
) -> Dict[str, Any]:
    """
    Convertit n'importe quelle exception en dictionnaire.

    Les exceptions SpeakerIdentity conservent leurs informations structurées.

    Les exceptions externes sont converties en erreur interne sans
    exposer leur message technique dans la réponse API/WebSocket.

    Parameters
    ----------
    exc:
        Exception à convertir.

    Returns
    -------
    dict
        Représentation sérialisable de l'erreur.
    """

    # ---------------------------------------------------------------
    # SpeakerIdentity EXCEPTION
    # ---------------------------------------------------------------

    if isinstance(
        exc,
        DiarizationError,
    ):
        return exc.to_dict()

    # ---------------------------------------------------------------
    # EXTERNAL EXCEPTION
    # ---------------------------------------------------------------

    exception_type = type(
        exc
    ).__name__

    if not exception_type:
        exception_type = "Exception"

    return {
        "error": "INTERNAL_ERROR",
        "message": (
            "An internal error occurred."
        ),
        "details": {
            "exception_type": exception_type,
        },
    }


# ====================================================================
# PUBLIC API
# ====================================================================


__all__ = [

    # ---------------------------------------------------------------
    # Root
    # ---------------------------------------------------------------

    "DiarizationError",

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------

    "ConfigurationError",

    # ---------------------------------------------------------------
    # Audio
    # ---------------------------------------------------------------

    "AudioError",
    "AudioFormatError",
    "AudioBufferError",
    "AudioProcessingError",

    # ---------------------------------------------------------------
    # VAD
    # ---------------------------------------------------------------

    "VADError",
    "VADConfigurationError",
    "VADProcessingError",

    # ---------------------------------------------------------------
    # Segmentation
    # ---------------------------------------------------------------

    "SegmentationError",
    "InvalidSegmentError",
    "SegmentLimitError",

    # ---------------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------------

    "EmbeddingError",
    "EmbeddingModelError",
    "EmbeddingDimensionError",

    # ---------------------------------------------------------------
    # Speakers
    # ---------------------------------------------------------------

    "SpeakerError",
    "SpeakerIdentificationError",
    "SpeakerMemoryError",
    "SpeakerLimitError",

    # ---------------------------------------------------------------
    # Sessions
    # ---------------------------------------------------------------

    "SessionError",
    "SessionNotFoundError",
    "SessionAlreadyExistsError",
    "SessionLimitError",
    "SessionStateError",
    "SessionIsolationError",

    # ---------------------------------------------------------------
    # Pipeline
    # ---------------------------------------------------------------

    "PipelineError",
    "PipelineStateError",
    "PipelineTimeoutError",

    # ---------------------------------------------------------------
    # WebSocket
    # ---------------------------------------------------------------

    "WebSocketError",
    "WebSocketProtocolError",
    "WebSocketPayloadError",

    # ---------------------------------------------------------------
    # API
    # ---------------------------------------------------------------

    "APIError",
    "APIValidationError",

    # ---------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------

    "exception_to_dict",
]