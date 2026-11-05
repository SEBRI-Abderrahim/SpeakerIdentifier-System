"""
====================================================================
SpeakerIdentity IA - Diarization Metrics
====================================================================

Projet       : SpeakerIdentity IA
Module       : Diarization
Fichier      : diarization/metrics.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 2.1.1
Date         : 14/08/2026

Description
------------

Monitoring centralisé du module de diarisation SpeakerIdentity IA.

Le système de métriques permet de suivre :

    - chunks audio reçus ;
    - volume audio reçu ;
    - frames traitées ;
    - frames vocales ;
    - frames silencieuses ;
    - segments créés ;
    - segments rejetés ;
    - speakers identifiés ;
    - speakers créés ;
    - sessions créées ;
    - sessions actives ;
    - erreurs ;
    - temps de traitement ;
    - latence de traitement ;
    - statistiques par session.

Contraintes
-----------

    - thread-safe ;
    - compatible multi-session ;
    - compatible temps réel ;
    - aucune dépendance externe ;
    - utilisable par API / WebSocket ;
    - compatible avec asyncio via appels synchrones courts.

Architecture
------------

    Audio
      |
      v
    Pipeline
      |
      +------> Metrics
      |
      +------> Health
      |
      +------> API
      |
      +------> WebSocket

====================================================================
"""

from __future__ import annotations

import math
import threading
import time

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ====================================================================
# INTERNAL VALIDATION HELPERS
# ====================================================================


def _validate_metric_name(
    field_name: str,
) -> str:
    """
    Valide le nom d'une métrique.
    """

    if not isinstance(field_name, str):
        raise TypeError(
            "metric field must be a string."
        )

    field_name = field_name.strip()

    if not field_name:
        raise ValueError(
            "metric field must not be empty."
        )

    return field_name


def _validate_session_id(
    session_id: str,
) -> str:
    """
    Normalise et valide un identifiant de session.
    """

    if session_id is None:
        raise ValueError(
            "session_id must not be empty."
        )

    session_id = str(
        session_id
    ).strip()

    if not session_id:
        raise ValueError(
            "session_id must not be empty."
        )

    return session_id


def _validate_non_negative_int(
    value: int,
    field_name: str,
) -> int:
    """
    Valide un entier >= 0.

    bool est explicitement refusé car bool
    est une sous-classe de int en Python.
    """

    if isinstance(value, bool):
        raise TypeError(
            f"{field_name} must be an integer."
        )

    if not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be an integer."
        )

    if value < 0:
        raise ValueError(
            f"{field_name} must be >= 0."
        )

    return value


def _validate_non_negative_float(
    value: float,
    field_name: str,
) -> float:
    """
    Valide une valeur numérique finie >= 0.
    """

    if isinstance(value, bool):
        raise TypeError(
            f"{field_name} must be a numeric value."
        )

    try:
        value = float(value)

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise TypeError(
            f"{field_name} must be a numeric value."
        ) from exc

    if not math.isfinite(value):

        raise ValueError(
            f"{field_name} must be finite."
        )

    if value < 0:

        raise ValueError(
            f"{field_name} must be >= 0."
        )

    return value


# ====================================================================
# GLOBAL METRICS SNAPSHOT
# ====================================================================


@dataclass
class MetricsSnapshot:
    """
    Instantané global des métriques du module.

    Cet objet est destiné à la lecture et à la sérialisation.
    Les modifications sont effectuées par ``DiarizationMetrics``.
    """

    # ---------------------------------------------------------------
    # Audio
    # ---------------------------------------------------------------

    audio_chunks: int = 0

    audio_bytes: int = 0

    # ---------------------------------------------------------------
    # Processing
    # ---------------------------------------------------------------

    frames_processed: int = 0

    speech_frames: int = 0

    silence_frames: int = 0

    # ---------------------------------------------------------------
    # Segmentation
    # ---------------------------------------------------------------

    segments_created: int = 0

    segments_rejected: int = 0

    # ---------------------------------------------------------------
    # Speakers
    # ---------------------------------------------------------------

    speakers_identified: int = 0

    speakers_created: int = 0

    # ---------------------------------------------------------------
    # Sessions
    # ---------------------------------------------------------------

    sessions_created: int = 0

    sessions_started: int = 0

    sessions_stopped: int = 0

    active_sessions: int = 0

    # ---------------------------------------------------------------
    # Errors
    # ---------------------------------------------------------------

    errors: int = 0

    # ---------------------------------------------------------------
    # Processing time
    # ---------------------------------------------------------------

    processing_time: float = 0.0

    total_latency: float = 0.0

    max_latency: float = 0.0

    latency_samples: int = 0

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    started_at: float = field(
        default_factory=time.time
    )

    # =================================================================
    # SERIALIZATION
    # =================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit le snapshot en dictionnaire.
        """

        now = time.time()

        uptime = max(
            0.0,
            now - self.started_at,
        )

        average_latency = 0.0

        if self.latency_samples > 0:

            average_latency = (
                self.total_latency
                / self.latency_samples
            )

        return {
            "audio_chunks": self.audio_chunks,

            "audio_bytes": self.audio_bytes,

            "frames_processed": (
                self.frames_processed
            ),

            "speech_frames": (
                self.speech_frames
            ),

            "silence_frames": (
                self.silence_frames
            ),

            "segments_created": (
                self.segments_created
            ),

            "segments_rejected": (
                self.segments_rejected
            ),

            "speakers_identified": (
                self.speakers_identified
            ),

            "speakers_created": (
                self.speakers_created
            ),

            "sessions_created": (
                self.sessions_created
            ),

            "sessions_started": (
                self.sessions_started
            ),

            "sessions_stopped": (
                self.sessions_stopped
            ),

            "active_sessions": (
                self.active_sessions
            ),

            "errors": self.errors,

            "processing_time": round(
                self.processing_time,
                6,
            ),

            "average_latency": round(
                average_latency,
                6,
            ),

            "max_latency": round(
                self.max_latency,
                6,
            ),

            "latency_samples": (
                self.latency_samples
            ),

            "uptime": round(
                uptime,
                3,
            ),

            "started_at": self.started_at,
        }


# ====================================================================
# SESSION METRICS
# ====================================================================


@dataclass
class SessionMetricsSnapshot:
    """
    Métriques spécifiques à une session.
    """

    session_id: str

    # ---------------------------------------------------------------
    # Session metadata
    # ---------------------------------------------------------------

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    # ---------------------------------------------------------------
    # Audio
    # ---------------------------------------------------------------

    audio_chunks: int = 0

    audio_bytes: int = 0

    # ---------------------------------------------------------------
    # Processing
    # ---------------------------------------------------------------

    frames_processed: int = 0

    speech_frames: int = 0

    silence_frames: int = 0

    # ---------------------------------------------------------------
    # Segmentation
    # ---------------------------------------------------------------

    segments_created: int = 0

    segments_rejected: int = 0

    # ---------------------------------------------------------------
    # Speakers
    # ---------------------------------------------------------------

    speakers_identified: int = 0

    speakers_created: int = 0

    # ---------------------------------------------------------------
    # Errors
    # ---------------------------------------------------------------

    errors: int = 0

    # ---------------------------------------------------------------
    # Processing time
    # ---------------------------------------------------------------

    processing_time: float = 0.0

    total_latency: float = 0.0

    max_latency: float = 0.0

    latency_samples: int = 0

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    started_at: float = field(
        default_factory=time.time
    )

    last_activity: float = field(
        default_factory=time.time
    )

    def __post_init__(self) -> None:
        """
        Valide l'identifiant de session.
        """

        self.session_id = _validate_session_id(
            self.session_id
        )

    # =================================================================
    # SERIALIZATION
    # =================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Sérialise les métriques de session.
        """

        average_latency = 0.0

        if self.latency_samples > 0:

            average_latency = (
                self.total_latency
                / self.latency_samples
            )

        return {
            "session_id": self.session_id,

            "audio_chunks": self.audio_chunks,

            "audio_bytes": self.audio_bytes,

            "frames_processed": (
                self.frames_processed
            ),

            "speech_frames": (
                self.speech_frames
            ),

            "silence_frames": (
                self.silence_frames
            ),

            "segments_created": (
                self.segments_created
            ),

            "segments_rejected": (
                self.segments_rejected
            ),

            "speakers_identified": (
                self.speakers_identified
            ),

            "speakers_created": (
                self.speakers_created
            ),

            "errors": self.errors,

            "processing_time": round(
                self.processing_time,
                6,
            ),

            "average_latency": round(
                average_latency,
                6,
            ),

            "max_latency": round(
                self.max_latency,
                6,
            ),

            "latency_samples": (
                self.latency_samples
            ),

            "started_at": self.started_at,

            "last_activity": (
                self.last_activity
            ),
        }


# ====================================================================
# DIARIZATION METRICS
# ====================================================================


class DiarizationMetrics:
    """
    Gestionnaire centralisé des métriques.

    Cette classe est thread-safe et supporte plusieurs
    sessions simultanées.

    Exemple :

        metrics.increment(
            "audio_chunks"
        )

        metrics.increment(
            "segments_created"
        )

        metrics.increment(
            "segments_rejected"
        )

        metrics.record_latency(
            0.124
        )
    """

    # =================================================================
    # INITIALIZATION
    # =================================================================

    def __init__(self) -> None:

        self._lock = threading.RLock()

        self.snapshot = MetricsSnapshot()

        self._sessions: Dict[
            str,
            SessionMetricsSnapshot,
        ] = {}

    # =================================================================
    # GENERIC INCREMENT
    # =================================================================

    def increment(
        self,
        field: str,
        value: int = 1,
        *,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Incrémente une métrique entière.

        Si ``session_id`` est fourni, la métrique
        globale ET celle de la session sont mises à jour.
        """

        field = _validate_metric_name(
            field
        )

        value = _validate_non_negative_int(
            value,
            "increment value",
        )

        normalized_session_id: Optional[str] = None

        if session_id is not None:

            normalized_session_id = (
                _validate_session_id(
                    session_id
                )
            )

        with self._lock:

            if not hasattr(
                self.snapshot,
                field,
            ):

                raise AttributeError(
                    f"Unknown metric: {field}"
                )

            current = getattr(
                self.snapshot,
                field,
            )

            if (
                isinstance(current, bool)
                or not isinstance(current, int)
            ):

                raise TypeError(
                    f"Metric '{field}' "
                    "is not an integer metric."
                )

            setattr(
                self.snapshot,
                field,
                current + value,
            )

            if normalized_session_id is not None:

                session = (
                    self._get_or_create_session(
                        normalized_session_id
                    )
                )

                if hasattr(
                    session,
                    field,
                ):

                    session_current = getattr(
                        session,
                        field,
                    )

                    if (
                        isinstance(
                            session_current,
                            bool,
                        )
                        or not isinstance(
                            session_current,
                            int,
                        )
                    ):

                        raise TypeError(
                            f"Metric '{field}' "
                            "is not an integer "
                            "session metric."
                        )

                    setattr(
                        session,
                        field,
                        session_current + value,
                    )

                session.last_activity = time.time()

    # =================================================================
    # ADD AUDIO BYTES
    # =================================================================

    def add_audio_bytes(
        self,
        value: int,
        *,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Ajoute un volume de données audio.
        """

        value = _validate_non_negative_int(
            value,
            "audio bytes",
        )

        self.increment(
            "audio_bytes",
            value,
            session_id=session_id,
        )

    # =================================================================
    # PROCESSING TIME
    # =================================================================

    def add_processing_time(
        self,
        seconds: float,
        *,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Ajoute du temps de traitement.
        """

        seconds = _validate_non_negative_float(
            seconds,
            "processing time",
        )

        normalized_session_id: Optional[str] = None

        if session_id is not None:

            normalized_session_id = (
                _validate_session_id(
                    session_id
                )
            )

        with self._lock:

            self.snapshot.processing_time += (
                seconds
            )

            if normalized_session_id is not None:

                session = (
                    self._get_or_create_session(
                        normalized_session_id
                    )
                )

                session.processing_time += (
                    seconds
                )

                session.last_activity = time.time()

    # =================================================================
    # LATENCY
    # =================================================================

    def record_latency(
        self,
        seconds: float,
        *,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Enregistre une latence de traitement.
        """

        seconds = _validate_non_negative_float(
            seconds,
            "latency",
        )

        normalized_session_id: Optional[str] = None

        if session_id is not None:

            normalized_session_id = (
                _validate_session_id(
                    session_id
                )
            )

        with self._lock:

            self.snapshot.total_latency += (
                seconds
            )

            self.snapshot.latency_samples += 1

            self.snapshot.max_latency = max(
                self.snapshot.max_latency,
                seconds,
            )

            if normalized_session_id is not None:

                session = (
                    self._get_or_create_session(
                        normalized_session_id
                    )
                )

                session.total_latency += (
                    seconds
                )

                session.latency_samples += 1

                session.max_latency = max(
                    session.max_latency,
                    seconds,
                )

                session.last_activity = time.time()

    # =================================================================
    # SESSION CREATED
    # =================================================================

    def session_created(
        self,
        session_id: str,
    ) -> None:
        """
        Enregistre la création d'une session.
        """

        session_id = _validate_session_id(
            session_id
        )

        with self._lock:

            session = self._sessions.get(
                session_id
            )

            if session is None:

                session = (
                    SessionMetricsSnapshot(
                        session_id=session_id
                    )
                )

                self._sessions[
                    session_id
                ] = session

            now = time.time()

            session.last_activity = now

            self.snapshot.sessions_created += 1

    # =================================================================
    # SESSION STARTED
    # =================================================================

    def session_started(
        self,
        session_id: str,
    ) -> None:
        """
        Enregistre le démarrage d'une session.
        """

        session_id = _validate_session_id(
            session_id
        )

        with self._lock:

            session = (
                self._get_or_create_session(
                    session_id
                )
            )

            now = time.time()

            was_active = bool(
                session.metadata.get(
                    "_metrics_active",
                    False,
                )
            )

            if not was_active:

                self.snapshot.sessions_started += 1

                self.snapshot.active_sessions += 1

                session.metadata[
                    "_metrics_active"
                ] = True

            session.last_activity = now

    # =================================================================
    # SESSION STOPPED
    # =================================================================

    def session_stopped(
        self,
        session_id: str,
    ) -> None:
        """
        Enregistre l'arrêt d'une session.
        """

        session_id = _validate_session_id(
            session_id
        )

        with self._lock:

            session = self._sessions.get(
                session_id
            )

            if session is None:
                return

            now = time.time()

            was_active = bool(
                session.metadata.get(
                    "_metrics_active",
                    False,
                )
            )

            if was_active:

                self.snapshot.sessions_stopped += 1

                self.snapshot.active_sessions = max(
                    0,
                    self.snapshot.active_sessions - 1,
                )

                session.metadata[
                    "_metrics_active"
                ] = False

            session.last_activity = now

    # =================================================================
    # ERROR
    # =================================================================

    def error(
        self,
        *,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Enregistre une erreur.
        """

        self.increment(
            "errors",
            1,
            session_id=session_id,
        )

    # =================================================================
    # SESSION ACCESS
    # =================================================================

    def _get_or_create_session(
        self,
        session_id: str,
    ) -> SessionMetricsSnapshot:
        """
        Retourne les métriques d'une session.

        Cette méthode doit être appelée avec le lock actif.
        """

        session_id = _validate_session_id(
            session_id
        )

        session = self._sessions.get(
            session_id
        )

        if session is None:

            session = SessionMetricsSnapshot(
                session_id=session_id
            )

            self._sessions[
                session_id
            ] = session

        return session

    # =================================================================
    # SNAPSHOT
    # =================================================================

    def snapshot_dict(self) -> Dict[str, Any]:
        """
        Retourne les métriques globales.
        """

        with self._lock:

            return self.snapshot.to_dict()

    # =================================================================
    # SESSION SNAPSHOT
    # =================================================================

    def session_snapshot(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:

        session_id = _validate_session_id(
            session_id
        )

        with self._lock:

            session = self._sessions.get(
                session_id
            )

            if session is None:
                return None

            return session.to_dict()

    # =================================================================
    # ALL SESSIONS
    # =================================================================

    def sessions_snapshot(
        self,
    ) -> Dict[
        str,
        Dict[str, Any],
    ]:

        with self._lock:

            return {
                session_id: session.to_dict()
                for (
                    session_id,
                    session,
                ) in self._sessions.items()
            }

    # =================================================================
    # SESSION REMOVAL
    # =================================================================

    def remove_session(
        self,
        session_id: str,
    ) -> bool:
        """
        Supprime les métriques d'une session.

        Retourne True si une session a été supprimée.
        """

        session_id = _validate_session_id(
            session_id
        )

        with self._lock:

            session = self._sessions.pop(
                session_id,
                None,
            )

            if session is None:
                return False

            was_active = bool(
                session.metadata.get(
                    "_metrics_active",
                    False,
                )
            )

            if was_active:

                self.snapshot.active_sessions = max(
                    0,
                    self.snapshot.active_sessions - 1,
                )

            return True

    # =================================================================
    # COMPLETE SNAPSHOT
    # =================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Retourne un snapshot complet.

        Format :

            {
                "global": {...},
                "sessions": {...}
            }
        """

        with self._lock:

            return {
                "global": (
                    self.snapshot.to_dict()
                ),

                "sessions": {
                    session_id: (
                        session.to_dict()
                    )
                    for (
                        session_id,
                        session,
                    ) in self._sessions.items()
                },
            }

    # =================================================================
    # RESET
    # =================================================================

    def reset(
        self,
        *,
        clear_sessions: bool = True,
    ) -> None:
        """
        Réinitialise les métriques.
        """

        with self._lock:

            self.snapshot = MetricsSnapshot()

            if clear_sessions:

                self._sessions.clear()


# ====================================================================
# GLOBAL INSTANCE
# ====================================================================


metrics = DiarizationMetrics()


# ====================================================================
# PUBLIC API
# ====================================================================


__all__ = [
    "MetricsSnapshot",
    "SessionMetricsSnapshot",
    "DiarizationMetrics",
    "metrics",
]