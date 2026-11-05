"""
====================================================================
SpeakerIdentity IA - Diarization Session Manager
====================================================================

Projet       : SpeakerIdentity IA
Module       : Diarization
Fichier      : diarization/session_manager.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 2.0.2
Date         : 14/08/2026

Description
------------

Gestionnaire central des sessions de diarisation SpeakerIdentity IA.

Le SessionManager assure uniquement la gestion du cycle de vie,
de l'isolation et des statistiques des sessions.

Il ne réalise aucun traitement audio, VAD, embedding, identification
speaker ou STT.

Format audio interne :

    PCM16
    16000 Hz
    Mono
    2 bytes/sample

====================================================================
"""

from __future__ import annotations

import threading
import time

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import (
    MAX_CONCURRENT_SESSIONS,
    MAX_SESSIONS,
    SESSION_MEMORY_TTL_SECONDS,
    SESSION_LOCK_TIMEOUT_SECONDS,
    MULTI_SESSION_ENABLED,
    SESSION_ISOLATION_ENABLED,
)

from .exceptions import (
    SessionError,
    SessionNotFoundError,
)

from .models import (
    AudioFormat,
    DiarizationSession,
    SessionStatus,
)


# ====================================================================
# SESSION CONTEXT
# ====================================================================


@dataclass
class SessionContext:
    """
    Contexte runtime isolé d'une session.
    """

    session: DiarizationSession
    created_at: float
    last_access: float
    processing_lock: threading.RLock
    metadata: Dict[str, Any]


# ====================================================================
# SESSION MANAGER
# ====================================================================


class DiarizationSessionManager:
    """
    Gestionnaire multi-session thread-safe.

    Chaque session possède son propre contexte et son propre verrou.
    """

    def __init__(
        self,
        *,
        max_sessions: int = MAX_SESSIONS,
        max_concurrent_sessions: int = MAX_CONCURRENT_SESSIONS,
        session_ttl_seconds: int = SESSION_MEMORY_TTL_SECONDS,
        lock_timeout_seconds: float = SESSION_LOCK_TIMEOUT_SECONDS,
        multi_session_enabled: bool = MULTI_SESSION_ENABLED,
        session_isolation_enabled: bool = SESSION_ISOLATION_ENABLED,
        audio_format: Optional[AudioFormat] = None,
        config: Optional[Any] = None,
    ) -> None:

        self.config = config

        self.max_sessions = self._validate_positive_int(
            max_sessions,
            "max_sessions",
        )

        self.max_concurrent_sessions = self._validate_positive_int(
            max_concurrent_sessions,
            "max_concurrent_sessions",
        )

        self.session_ttl_seconds = self._validate_non_negative_int(
            session_ttl_seconds,
            "session_ttl_seconds",
        )

        try:
            self.lock_timeout_seconds = float(
                lock_timeout_seconds
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "lock_timeout_seconds must be numeric."
            ) from exc

        if self.lock_timeout_seconds <= 0:
            raise ValueError(
                "lock_timeout_seconds must be > 0."
            )

        self.multi_session_enabled = bool(
            multi_session_enabled
        )

        self.session_isolation_enabled = bool(
            session_isolation_enabled
        )

        self.audio_format = (
            audio_format
            if audio_format is not None
            else AudioFormat()
        )

        self._sessions: Dict[
            str,
            SessionContext,
        ] = {}

        self._lock = threading.RLock()

        self._created_count = 0
        self._started_count = 0
        self._stopped_count = 0
        self._removed_count = 0
        self._expired_count = 0

        self._created_at = time.time()

    # =================================================================
    # VALIDATION HELPERS
    # =================================================================

    @staticmethod
    def _validate_positive_int(
        value: Any,
        name: str,
    ) -> int:

        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be an integer."
            ) from exc

        if value < 1:
            raise ValueError(
                f"{name} must be >= 1."
            )

        return value

    @staticmethod
    def _validate_non_negative_int(
        value: Any,
        name: str,
    ) -> int:

        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be an integer."
            ) from exc

        if value < 0:
            raise ValueError(
                f"{name} must be >= 0."
            )

        return value

    # =================================================================
    # SESSION ID
    # =================================================================

    @staticmethod
    def _validate_session_id(
        session_id: str,
    ) -> str:

        if session_id is None:
            raise SessionError(
                "session_id cannot be None."
            )

        session_id = str(
            session_id
        ).strip()

        if not session_id:
            raise SessionError(
                "session_id cannot be empty."
            )

        if len(session_id) > 256:
            raise SessionError(
                "session_id is too long."
            )

        return session_id

    # =================================================================
    # CREATE
    # =================================================================

    def create_session(
        self,
        session_id: Optional[str] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        audio_format: Optional[AudioFormat] = None,
    ) -> DiarizationSession:

        with self._lock:

            self._cleanup_expired_locked()

            if not self.multi_session_enabled and self._sessions:
                raise SessionError(
                    "Multi-session is disabled. "
                    "Only one session is allowed."
                )

            if session_id is None:
                session_id = self._generate_session_id()
            else:
                session_id = self._validate_session_id(
                    session_id
                )

            if session_id in self._sessions:
                raise SessionError(
                    f"Session already exists: {session_id}"
                )

            if len(self._sessions) >= self.max_sessions:
                raise SessionError(
                    "Maximum number of sessions reached."
                )

            active_count = self._active_count_locked()

            if (
                self.multi_session_enabled
                and active_count >= self.max_concurrent_sessions
            ):
                raise SessionError(
                    "Maximum number of concurrent "
                    "sessions reached."
                )

            session_audio_format = (
                audio_format
                if audio_format is not None
                else self.audio_format
            )

            session_metadata = dict(
                metadata or {}
            )

            session = DiarizationSession(
                session_id=session_id,
                audio_format=session_audio_format,
                metadata=session_metadata,
            )

            now = time.time()

            runtime_metadata = dict(
                session_metadata
            )

            runtime_metadata.setdefault(
                "audio_stats",
                {
                    "audio_duration": 0.0,
                    "speech_duration": 0.0,
                    "audio_chunks": 0,
                },
            )

            runtime_metadata.setdefault(
                "speakers",
                {},
            )

            runtime_metadata.setdefault(
                "speaker_ids",
                [],
            )

            self._sessions[session_id] = SessionContext(
                session=session,
                created_at=now,
                last_access=now,
                processing_lock=threading.RLock(),
                metadata=runtime_metadata,
            )

            self._created_count += 1

            return session

    # =================================================================
    # GENERATE ID
    # =================================================================

    @staticmethod
    def _generate_session_id() -> str:

        timestamp = int(
            time.time() * 1000
        )

        suffix = time.time_ns() % 1_000_000

        return (
            f"session_{timestamp}_{suffix:06d}"
        )

    # =================================================================
    # GET
    # =================================================================

    def get_session(
        self,
        session_id: str,
        *,
        touch: bool = True,
    ) -> DiarizationSession:

        return self.get_context(
            session_id,
            touch=touch,
        ).session

    # =================================================================
    # GET CONTEXT
    # =================================================================

    def get_context(
        self,
        session_id: str,
        *,
        touch: bool = True,
    ) -> SessionContext:

        session_id = self._validate_session_id(
            session_id
        )

        with self._lock:

            context = self._sessions.get(
                session_id
            )

            if context is None:
                raise SessionNotFoundError(
                    f"Session not found: {session_id}"
                )

            if touch:
                now = time.time()

                context.last_access = now

                if hasattr(
                    context.session,
                    "last_activity",
                ):
                    context.session.last_activity = now

            return context

    # =================================================================
    # EXISTS
    # =================================================================

    def exists(
        self,
        session_id: str,
    ) -> bool:

        session_id = self._validate_session_id(
            session_id
        )

        with self._lock:
            return session_id in self._sessions

    # =================================================================
    # START
    # =================================================================

    def start_session(
        self,
        session_id: str,
    ) -> DiarizationSession:

        context = self.get_context(
            session_id
        )

        with context.processing_lock:

            session = context.session

            if session.status == SessionStatus.STOPPED:
                raise SessionError(
                    f"Cannot restart stopped session: "
                    f"{session_id}"
                )

            if session.status == SessionStatus.ERROR:
                raise SessionError(
                    f"Cannot start session in ERROR state: "
                    f"{session_id}"
                )

            if session.status != SessionStatus.RUNNING:

                with self._lock:

                    active_count = self._active_count_locked()

                    if active_count >= self.max_concurrent_sessions:
                        raise SessionError(
                            "Maximum number of concurrent "
                            "sessions reached."
                        )

                    session.start()

                    self._started_count += 1

            self._touch_context(
                context
            )

            return session

    # =================================================================
    # PAUSE
    # =================================================================

    def pause_session(
        self,
        session_id: str,
    ) -> DiarizationSession:

        context = self.get_context(
            session_id
        )

        with context.processing_lock:

            session = context.session

            if session.status == SessionStatus.STOPPED:
                raise SessionError(
                    f"Cannot pause stopped session: "
                    f"{session_id}"
                )

            session.pause()

            self._touch_context(
                context
            )

            return session

    # =================================================================
    # RESUME
    # =================================================================

    def resume_session(
        self,
        session_id: str,
    ) -> DiarizationSession:

        context = self.get_context(
            session_id
        )

        with context.processing_lock:

            session = context.session

            if session.status == SessionStatus.STOPPED:
                raise SessionError(
                    f"Cannot resume stopped session: "
                    f"{session_id}"
                )

            if session.status == SessionStatus.ERROR:
                raise SessionError(
                    f"Cannot resume session in ERROR state: "
                    f"{session_id}"
                )

            if session.status != SessionStatus.RUNNING:

                with self._lock:

                    if (
                        self._active_count_locked()
                        >= self.max_concurrent_sessions
                    ):
                        raise SessionError(
                            "Maximum number of concurrent "
                            "sessions reached."
                        )

                    if hasattr(
                        session,
                        "resume",
                    ):
                        session.resume()
                    else:
                        session.start()

                    self._started_count += 1

            self._touch_context(
                context
            )

            return session

    # =================================================================
    # STOP
    # =================================================================

    def stop_session(
        self,
        session_id: str,
    ) -> DiarizationSession:

        context = self.get_context(
            session_id
        )

        with context.processing_lock:

            session = context.session

            if session.status != SessionStatus.STOPPED:

                session.stop()

                with self._lock:
                    self._stopped_count += 1

            self._touch_context(
                context
            )

            return session

    # =================================================================
    # ERROR
    # =================================================================

    def mark_error(
        self,
        session_id: str,
    ) -> DiarizationSession:

        context = self.get_context(
            session_id
        )

        with context.processing_lock:

            context.session.set_error()

            self._touch_context(
                context
            )

            return context.session

    # =================================================================
    # AUDIO STATS
    # =================================================================

    def update_audio_stats(
        self,
        session_id: str,
        *,
        duration: float = 0.0,
        speech_duration: float = 0.0,
    ) -> DiarizationSession:

        session_id = self._validate_session_id(
            session_id
        )

        try:
            duration = float(duration)
            speech_duration = float(speech_duration)
        except (TypeError, ValueError) as exc:
            raise SessionError(
                "Audio durations must be numeric."
            ) from exc

        if duration < 0:
            raise SessionError(
                "duration must be >= 0."
            )

        if speech_duration < 0:
            raise SessionError(
                "speech_duration must be >= 0."
            )

        if speech_duration > duration:
            raise SessionError(
                "speech_duration cannot be greater "
                "than duration."
            )

        context = self.get_context(
            session_id,
            touch=False,
        )

        with context.processing_lock:

            stats = context.metadata.setdefault(
                "audio_stats",
                {
                    "audio_duration": 0.0,
                    "speech_duration": 0.0,
                    "audio_chunks": 0,
                },
            )

            stats["audio_duration"] += duration
            stats["speech_duration"] += speech_duration
            stats["audio_chunks"] += 1

            session = context.session

            if hasattr(
                session,
                "audio_duration",
            ):
                session.audio_duration += duration

            if hasattr(
                session,
                "speech_duration",
            ):
                session.speech_duration += speech_duration

            if hasattr(
                session,
                "audio_chunks",
            ):
                session.audio_chunks += 1

            self._touch_context(
                context
            )

            return session

    # =================================================================
    # ASSIGN SPEAKER
    # =================================================================

    def assign_speaker(
        self,
        session_id: str,
        speaker_id: str,
        segment: Optional[Any] = None,
    ) -> DiarizationSession:

        session_id = self._validate_session_id(
            session_id
        )

        if speaker_id is None:
            raise SessionError(
                "speaker_id cannot be None."
            )

        speaker_id = str(
            speaker_id
        ).strip()

        if not speaker_id:
            raise SessionError(
                "speaker_id cannot be empty."
            )

        context = self.get_context(
            session_id,
            touch=False,
        )

        with context.processing_lock:

            session = context.session

            if segment is not None:

                if hasattr(
                    segment,
                    "speaker_id",
                ):
                    segment.speaker_id = speaker_id

                elif isinstance(
                    segment,
                    dict,
                ):
                    segment["speaker_id"] = speaker_id

            speakers = context.metadata.setdefault(
                "speakers",
                {},
            )

            now = time.time()

            speaker_entry = speakers.get(
                speaker_id
            )

            if speaker_entry is None:

                speaker_entry = {
                    "speaker_id": speaker_id,
                    "segment_count": 0,
                    "first_seen": now,
                    "last_seen": now,
                }

                speakers[speaker_id] = speaker_entry

            else:
                speaker_entry["last_seen"] = now

            if segment is not None:
                speaker_entry["segment_count"] += 1

            speaker_ids = context.metadata.setdefault(
                "speaker_ids",
                [],
            )

            if speaker_id not in speaker_ids:
                speaker_ids.append(
                    speaker_id
                )

            if hasattr(
                session,
                "speaker_ids",
            ):

                try:
                    if session.speaker_ids is None:
                        session.speaker_ids = []

                    if speaker_id not in session.speaker_ids:
                        session.speaker_ids.append(
                            speaker_id
                        )
                except (AttributeError, TypeError):
                    pass

            if hasattr(
                session,
                "speakers",
            ):

                try:
                    if session.speakers is None:
                        session.speakers = {}

                    session.speakers.setdefault(
                        speaker_id,
                        [],
                    )

                    if segment is not None:
                        session.speakers[
                            speaker_id
                        ].append(segment)

                except (AttributeError, TypeError):
                    pass

            self._touch_context(
                context
            )

            return session

    # =================================================================
    # GET SPEAKERS
    # =================================================================

    def get_speakers(
        self,
        session_id: str,
    ) -> List[str]:

        context = self.get_context(
            session_id,
            touch=False,
        )

        with context.processing_lock:

            return list(
                context.metadata.get(
                    "speaker_ids",
                    [],
                )
            )

    # =================================================================
    # AUDIO STATISTICS
    # =================================================================

    def get_audio_stats(
        self,
        session_id: str,
    ) -> Dict[str, Any]:

        context = self.get_context(
            session_id,
            touch=False,
        )

        with context.processing_lock:

            return dict(
                context.metadata.get(
                    "audio_stats",
                    {},
                )
            )

    # =================================================================
    # REMOVE
    # =================================================================

    def remove_session(
        self,
        session_id: str,
        *,
        ignore_missing: bool = False,
    ) -> bool:

        session_id = self._validate_session_id(
            session_id
        )

        # Ne pas conserver _lock pendant l'acquisition de
        # processing_lock afin d'éviter une inversion de verrous
        # avec start_session(), resume_session() et les autres
        # méthodes utilisant processing_lock puis _lock.
        with self._lock:

            context = self._sessions.get(
                session_id
            )

        if context is None:

            if ignore_missing:
                return False

            raise SessionNotFoundError(
                f"Session not found: {session_id}"
            )

        acquired = context.processing_lock.acquire(
            timeout=self.lock_timeout_seconds
        )

        if not acquired:
            raise SessionError(
                f"Unable to acquire processing lock "
                f"for session: {session_id}"
            )

        try:

            with self._lock:

                current_context = self._sessions.get(
                    session_id
                )

                if current_context is None:

                    if ignore_missing:
                        return False

                    raise SessionNotFoundError(
                        f"Session not found: {session_id}"
                    )

                if current_context is not context:
                    raise SessionError(
                        f"Session context changed unexpectedly: "
                        f"{session_id}"
                    )

                self._sessions.pop(
                    session_id,
                    None,
                )

                self._removed_count += 1

                return True

        finally:
            context.processing_lock.release()

    # =================================================================
    # CLEAR
    # =================================================================

    def clear(
        self,
        *,
        include_running: bool = False,
    ) -> int:

        removed = 0

        with self._lock:
            session_items = list(
                self._sessions.items()
            )

        for session_id, context in session_items:

            if (
                not include_running
                and context.session.status
                == SessionStatus.RUNNING
            ):
                continue

            acquired = context.processing_lock.acquire(
                timeout=self.lock_timeout_seconds
            )

            if not acquired:
                continue

            try:

                with self._lock:

                    current_context = self._sessions.get(
                        session_id
                    )

                    if current_context is not context:
                        continue

                    self._sessions.pop(
                        session_id,
                        None,
                    )

                    removed += 1
                    self._removed_count += 1

            finally:
                context.processing_lock.release()

        return removed

    # =================================================================
    # LIST
    # =================================================================

    def list_sessions(
        self,
        *,
        include_stopped: bool = True,
    ) -> List[DiarizationSession]:

        with self._lock:

            result = []

            for context in self._sessions.values():

                session = context.session

                if (
                    not include_stopped
                    and session.status
                    == SessionStatus.STOPPED
                ):
                    continue

                result.append(
                    session
                )

            return result

    # =================================================================
    # SESSION IDS
    # =================================================================

    def session_ids(self) -> List[str]:

        with self._lock:
            return list(
                self._sessions.keys()
            )

    # =================================================================
    # ACTIVE COUNT
    # =================================================================

    @property
    def active_count(self) -> int:

        with self._lock:
            return self._active_count_locked()

    def _active_count_locked(self) -> int:

        return sum(
            1
            for context in self._sessions.values()
            if context.session.status
            == SessionStatus.RUNNING
        )

    # =================================================================
    # TOTAL COUNT
    # =================================================================

    @property
    def count(self) -> int:

        with self._lock:
            return len(
                self._sessions
            )

    # =================================================================
    # PROCESSING LOCK
    # =================================================================

    def acquire_session(
        self,
        session_id: str,
        *,
        blocking: bool = True,
    ):

        context = self.get_context(
            session_id
        )

        if not blocking:

            acquired = context.processing_lock.acquire(
                blocking=False
            )

        else:

            acquired = context.processing_lock.acquire(
                timeout=self.lock_timeout_seconds
            )

        if not acquired:
            raise SessionError(
                f"Unable to acquire processing lock "
                f"for session: {session_id}"
            )

        return context.processing_lock

    # =================================================================
    # TOUCH
    # =================================================================

    def touch(
        self,
        session_id: str,
    ) -> None:

        context = self.get_context(
            session_id,
            touch=False,
        )

        self._touch_context(
            context
        )

    def _touch_context(
        self,
        context: SessionContext,
    ) -> None:

        now = time.time()

        context.last_access = now

        if hasattr(
            context.session,
            "last_activity",
        ):
            context.session.last_activity = now

    # =================================================================
    # EXPIRATION
    # =================================================================

    def is_expired(
        self,
        context: SessionContext,
        *,
        now: Optional[float] = None,
    ) -> bool:

        if self.session_ttl_seconds <= 0:
            return False

        if now is None:
            now = time.time()
        else:
            now = float(now)

        if context.session.status == SessionStatus.RUNNING:
            return False

        return (
            now - context.last_access
            >= self.session_ttl_seconds
        )

    # =================================================================
    # CLEANUP INTERNAL
    # =================================================================

    def _cleanup_expired_locked(self) -> int:

        if self.session_ttl_seconds <= 0:
            return 0

        now = time.time()

        expired_ids = [
            session_id
            for session_id, context
            in self._sessions.items()
            if self.is_expired(
                context,
                now=now,
            )
        ]

        removed = 0

        for session_id in expired_ids:

            context = self._sessions.get(
                session_id
            )

            if context is None:
                continue

            acquired = context.processing_lock.acquire(
                timeout=self.lock_timeout_seconds
            )

            if not acquired:
                continue

            try:

                if session_id not in self._sessions:
                    continue

                self._sessions.pop(
                    session_id,
                    None,
                )

                self._removed_count += 1
                self._expired_count += 1

                removed += 1

            finally:
                context.processing_lock.release()

        return removed

    # =================================================================
    # CLEANUP PUBLIC
    # =================================================================

    def cleanup_expired(self) -> int:

        with self._lock:
            return self._cleanup_expired_locked()

    # =================================================================
    # STATISTICS
    # =================================================================

    def statistics(self) -> Dict[str, Any]:

        with self._lock:

            status_counts = {
                status.value: 0
                for status in SessionStatus
            }

            total_audio_duration = 0.0
            total_speech_duration = 0.0
            total_audio_chunks = 0
            total_speakers = 0

            for context in self._sessions.values():

                session = context.session

                status_counts[
                    session.status.value
                ] += 1

                audio_stats = context.metadata.get(
                    "audio_stats",
                    {},
                )

                total_audio_duration += float(
                    audio_stats.get(
                        "audio_duration",
                        0.0,
                    )
                )

                total_speech_duration += float(
                    audio_stats.get(
                        "speech_duration",
                        0.0,
                    )
                )

                total_audio_chunks += int(
                    audio_stats.get(
                        "audio_chunks",
                        0,
                    )
                )

                total_speakers += len(
                    context.metadata.get(
                        "speaker_ids",
                        [],
                    )
                )

            return {
                "total_sessions": len(
                    self._sessions
                ),
                "active_sessions": self._active_count_locked(),
                "max_sessions": self.max_sessions,
                "max_concurrent_sessions": (
                    self.max_concurrent_sessions
                ),
                "multi_session_enabled": (
                    self.multi_session_enabled
                ),
                "session_isolation_enabled": (
                    self.session_isolation_enabled
                ),
                "session_ttl_seconds": (
                    self.session_ttl_seconds
                ),
                "created_count": self._created_count,
                "started_count": self._started_count,
                "stopped_count": self._stopped_count,
                "removed_count": self._removed_count,
                "expired_count": self._expired_count,
                "audio_duration": round(
                    total_audio_duration,
                    3,
                ),
                "speech_duration": round(
                    total_speech_duration,
                    3,
                ),
                "audio_chunks": total_audio_chunks,
                "speakers": total_speakers,
                "statuses": status_counts,
                "uptime": round(
                    time.time() - self._created_at,
                    3,
                ),
            }

    # =================================================================
    # HEALTH
    # =================================================================

    def health(self) -> Dict[str, Any]:

        with self._lock:

            total_sessions = len(
                self._sessions
            )

            active_sessions = self._active_count_locked()

            healthy = (
                total_sessions <= self.max_sessions
                and active_sessions <= self.max_concurrent_sessions
            )

            return {
                "status": (
                    "healthy"
                    if healthy
                    else "degraded"
                ),
                "component": (
                    "diarization_session_manager"
                ),
                "sessions": total_sessions,
                "active_sessions": active_sessions,
                "max_sessions": self.max_sessions,
                "max_concurrent_sessions": (
                    self.max_concurrent_sessions
                ),
                "multi_session_enabled": (
                    self.multi_session_enabled
                ),
                "session_isolation_enabled": (
                    self.session_isolation_enabled
                ),
            }

    # =================================================================
    # SERIALIZATION
    # =================================================================

    def to_dict(
        self,
        *,
        include_sessions: bool = True,
        include_segments: bool = False,
    ) -> Dict[str, Any]:

        result = {
            "statistics": self.statistics(),
            "health": self.health(),
        }

        if include_sessions:

            sessions = []

            for session in self.list_sessions():

                if hasattr(
                    session,
                    "to_dict",
                ):

                    sessions.append(
                        session.to_dict(
                            include_segments=include_segments
                        )
                    )

                else:

                    sessions.append(
                        {
                            "session_id": getattr(
                                session,
                                "session_id",
                                None,
                            ),
                            "status": str(
                                getattr(
                                    session,
                                    "status",
                                    "unknown",
                                )
                            ),
                        }
                    )

            result["sessions"] = sessions

        return result

    # =================================================================
    # SHUTDOWN
    # =================================================================

    def shutdown(self) -> None:

        with self._lock:

            contexts = list(
                self._sessions.values()
            )

        for context in contexts:

            acquired = context.processing_lock.acquire(
                timeout=self.lock_timeout_seconds
            )

            if not acquired:
                continue

            try:

                if (
                    context.session.status
                    == SessionStatus.RUNNING
                ):

                    context.session.stop()

                    with self._lock:
                        self._stopped_count += 1

                    self._touch_context(
                        context
                    )

            finally:
                context.processing_lock.release()


# ====================================================================
# GLOBAL MANAGER
# ====================================================================

session_manager = DiarizationSessionManager()

diarization_session_manager = session_manager


# ====================================================================
# PUBLIC API
# ====================================================================

__all__ = [
    "SessionContext",
    "DiarizationSessionManager",
    "session_manager",
    "diarization_session_manager",
    "SessionManager",
]


# ====================================================================
# ALIAS
# ====================================================================
#
# Meme raison que dans pipeline.py : le nom court est celui utilise
# par les tests et par le serveur de demonstration.

SessionManager = DiarizationSessionManager