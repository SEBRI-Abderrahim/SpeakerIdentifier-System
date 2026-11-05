"""
====================================================================
SpeakerIdentity IA - Diarization Engine
====================================================================

Projet       : SpeakerIdentity IA
Module       : Diarization
Fichier      : diarization/diarization_engine.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 2.0.1
Date         : 14/08/2026

Description
------------

Façade principale du système de diarisation SpeakerIdentity IA.

Responsabilités :

    - gestion du cycle de vie du moteur ;
    - création et gestion des sessions ;
    - réception des données PCM16 ;
    - orchestration du pipeline ;
    - identification des speakers ;
    - production des résultats ;
    - production des événements ;
    - support multi-session ;
    - support temps réel ;
    - health check ;
    - statistiques ;
    - préparation de l'intégration STT.

Format audio interne :

    Codec       : PCM16
    Sample rate : 16000 Hz
    Channels    : 1
    Width       : 2 bytes

====================================================================
"""

from __future__ import annotations

import threading
import time

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from .config import (
    DiarizationConfig,
    config as default_config,
)

from .exceptions import (
    AudioFormatError,
    DiarizationError,
    PipelineError,
    SessionError,
    SessionNotFoundError,
)

from .metrics import (
    DiarizationMetrics,
    metrics as default_metrics,
)

from .health import (
    DiarizationHealth,
)

from .pipeline import (
    DiarizationPipeline,
    PipelineResult,
    create_pipeline,
)

from .session_manager import (
    DiarizationSessionManager,
)


# ====================================================================
# ENGINE STATE
# ====================================================================

class EngineState:

    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


# ====================================================================
# ENGINE
# ====================================================================

class DiarizationEngine:

    def __init__(
        self,
        config: Optional[
            DiarizationConfig
        ] = None,
        pipeline: Optional[
            DiarizationPipeline
        ] = None,
        session_manager: Optional[
            DiarizationSessionManager
        ] = None,
        metrics: Optional[
            DiarizationMetrics
        ] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else default_config
        )

        self.config.validate()

        self.metrics = (
            metrics
            if metrics is not None
            else default_metrics
        )

        self.session_manager = (
            session_manager
            if session_manager is not None
            else DiarizationSessionManager(
                config=self.config
            )
        )

        # -------------------------------------------------------------
        # CORRECTIF : un seul gestionnaire de sessions
        # -------------------------------------------------------------
        #
        # Le moteur construisait SON gestionnaire de sessions, puis
        # create_pipeline() en construisait un SECOND, independant.
        # Les sessions etaient creees via self.pipeline.start_session(),
        # donc enregistrees dans celui du pipeline, mais relues via
        # self.session_manager, qui restait vide.
        #
        # Consequence : create_session() renvoyait bien une session,
        # mais get_session(), list_sessions(), session_count(),
        # pause/resume/stop/delete_session et process_audio echouaient
        # tous avec SessionNotFoundError. Verifie par reproduction :
        # create_session("s1") puis session_count() renvoyait 0.
        #
        # Le pipeline recoit desormais le gestionnaire du moteur ; et si
        # un pipeline est fourni de l'exterieur, c'est le sien qui fait
        # foi. Dans les deux cas il n'y en a plus qu'un.

        if pipeline is not None:

            self.pipeline = pipeline

            pipeline_manager = getattr(
                pipeline,
                "session_manager",
                None,
            )

            if pipeline_manager is not None:

                self.session_manager = pipeline_manager

        else:

            self.pipeline = DiarizationPipeline(
                config=self.config,
                session_manager=self.session_manager,
            )

        self.health_monitor = (
            DiarizationHealth()
        )

        self.health_monitor.register(
            "pipeline",
            self.pipeline,
        )

        self.health_monitor.register(
            "session_manager",
            self.session_manager,
        )

        self._state = (
            EngineState.CREATED
        )

        self._lock = threading.RLock()

        self._started_at: Optional[
            float
        ] = None

        self._stopped_at: Optional[
            float
        ] = None

    # =================================================================
    # STATE
    # =================================================================

    @property
    def state(self) -> str:

        with self._lock:
            return self._state

    @property
    def running(self) -> bool:

        return (
            self.state
            == EngineState.RUNNING
        )

    # =================================================================
    # START
    # =================================================================

    def start(self) -> None:

        with self._lock:

            if (
                self._state
                == EngineState.RUNNING
            ):
                return

            if (
                self._state
                == EngineState.ERROR
            ):

                raise PipelineError(
                    "DiarizationEngine is in error state."
                )

            try:

                self._started_at = time.time()

                self._stopped_at = None

                self._state = (
                    EngineState.RUNNING
                )

            except Exception as exc:

                self._state = (
                    EngineState.ERROR
                )

                self.metrics.increment(
                    "errors"
                )

                raise PipelineError(
                    f"Unable to start diarization engine: "
                    f"{exc}"
                ) from exc

    # =================================================================
    # STOP
    # =================================================================

    def stop(
        self,
        stop_sessions: bool = True,
    ) -> None:

        with self._lock:

            if (
                self._state
                == EngineState.STOPPED
            ):
                return

            try:

                if stop_sessions:

                    self._stop_all_sessions()

                self._stopped_at = time.time()

                self._state = (
                    EngineState.STOPPED
                )

            except Exception as exc:

                self._state = (
                    EngineState.ERROR
                )

                self.metrics.increment(
                    "errors"
                )

                raise PipelineError(
                    f"Unable to stop diarization engine: "
                    f"{exc}"
                ) from exc

    # =================================================================
    # CREATE SESSION
    # =================================================================

    def create_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Any:

        self._ensure_running()

        if session_id is not None:

            session_id = self._validate_session_id(
                session_id
            )

            # ---------------------------------------------------------
            # SESSION ALREADY EXISTS
            # ---------------------------------------------------------

            try:

                existing = (
                    self.session_manager.get_session(
                        session_id
                    )
                )

                if existing is not None:

                    return existing

            except Exception:
                pass

        try:

            session = (
                self.pipeline.start_session(
                    session_id=session_id,
                    metadata=metadata,
                )
            )

            return session

        except (
            SessionNotFoundError,
            SessionError,
            DiarizationError,
        ):

            raise

        except Exception as exc:

            self.metrics.increment(
                "errors"
            )

            raise SessionError(
                f"Unable to create session: {exc}"
            ) from exc

    # =================================================================
    # START SESSION
    # =================================================================

    def start_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Any:

        return self.create_session(
            session_id=session_id,
            metadata=metadata,
        )

    # =================================================================
    # GET SESSION
    # =================================================================

    def get_session(
        self,
        session_id: str,
    ) -> Any:

        session_id = self._validate_session_id(
            session_id
        )

        session = (
            self.session_manager.get_session(
                session_id
            )
        )

        if session is None:

            raise SessionNotFoundError(
                f"Session not found: {session_id}"
            )

        return session

    # =================================================================
    # PAUSE
    # =================================================================

    def pause_session(
        self,
        session_id: str,
    ) -> Any:

        self._ensure_running()

        session_id = self._validate_session_id(
            session_id
        )

        return self._call_session_operation(
            session_id,
            (
                "pause_session",
                "pause",
            ),
        )

    # =================================================================
    # RESUME
    # =================================================================

    def resume_session(
        self,
        session_id: str,
    ) -> Any:

        self._ensure_running()

        session_id = self._validate_session_id(
            session_id
        )

        return self._call_session_operation(
            session_id,
            (
                "resume_session",
                "resume",
            ),
        )

    # =================================================================
    # STOP SESSION
    # =================================================================

    def stop_session(
        self,
        session_id: str,
    ) -> Dict[str, Any]:

        self._ensure_running()

        session_id = self._validate_session_id(
            session_id
        )

        try:

            # ---------------------------------------------------------
            # FLUSH
            # ---------------------------------------------------------

            flush_result = self.pipeline.flush(
                session_id
            )

            # ---------------------------------------------------------
            # STOP
            # ---------------------------------------------------------

            session = (
                self.pipeline.stop_session(
                    session_id
                )
            )

            return {
                "session": session,
                "result": flush_result,
            }

        except (
            SessionNotFoundError,
            SessionError,
            DiarizationError,
        ):

            raise

        except Exception as exc:

            self.metrics.increment(
                "errors"
            )

            raise SessionError(
                f"Unable to stop session "
                f"{session_id}: {exc}"
            ) from exc

    # =================================================================
    # DELETE SESSION
    # =================================================================

    def delete_session(
        self,
        session_id: str,
    ) -> Any:

        self._ensure_running()

        session_id = self._validate_session_id(
            session_id
        )

        try:

            method = getattr(
                self.pipeline,
                "delete_session",
                None,
            )

            if method is not None:

                return method(
                    session_id
                )

            # CORRECTIF : la methode du gestionnaire de sessions
            # s'appelle remove_session(), pas delete_session(). Le
            # moteur cherchait un nom qui n'existe nulle part et levait
            # systematiquement "No delete_session() method available."
            for nom in (
                "delete_session",
                "remove_session",
            ):

                method = getattr(
                    self.session_manager,
                    nom,
                    None,
                )

                if method is not None:

                    return method(
                        session_id
                    )

            raise SessionError(
                "No delete_session() method available."
            )

        except (
            SessionNotFoundError,
            SessionError,
            DiarizationError,
        ):

            raise

        except Exception as exc:

            self.metrics.increment(
                "errors"
            )

            raise SessionError(
                f"Unable to delete session "
                f"{session_id}: {exc}"
            ) from exc

    # =================================================================
    # PROCESS AUDIO
    # =================================================================

    def process_audio(
        self,
        session_id: str,
        audio: bytes,
    ) -> PipelineResult:

        self._ensure_running()

        session_id = self._validate_session_id(
            session_id
        )

        if audio is None:

            raise AudioFormatError(
                "Audio cannot be None."
            )

        if not isinstance(
            audio,
            (
                bytes,
                bytearray,
            ),
        ):

            raise AudioFormatError(
                "Audio must be bytes or bytearray."
            )

        audio = bytes(audio)

        if not audio:

            return PipelineResult(
                session_id=session_id,
                segments=[],
                events=[],
            )

        try:

            # ---------------------------------------------------------
            # ENSURE SESSION
            # ---------------------------------------------------------

            self.get_session(
                session_id
            )

            # ---------------------------------------------------------
            # PROCESS
            # ---------------------------------------------------------

            result = self.pipeline.process(
                session_id=session_id,
                audio=audio,
            )

            return result

        except (
            SessionNotFoundError,
            SessionError,
            AudioFormatError,
            DiarizationError,
        ):

            raise

        except Exception as exc:

            self.metrics.increment(
                "errors"
            )

            raise PipelineError(
                f"Engine audio processing failed: "
                f"{exc}"
            ) from exc

    # =================================================================
    # PROCESS ALIAS
    # =================================================================

    def process(
        self,
        session_id: str,
        audio: bytes,
    ) -> PipelineResult:

        return self.process_audio(
            session_id=session_id,
            audio=audio,
        )

    # =================================================================
    # FLUSH
    # =================================================================

    def flush(
        self,
        session_id: str,
    ) -> PipelineResult:

        self._ensure_running()

        session_id = self._validate_session_id(
            session_id
        )

        try:

            self.get_session(
                session_id
            )

            return self.pipeline.flush(
                session_id
            )

        except (
            SessionNotFoundError,
            SessionError,
            DiarizationError,
        ):

            raise

        except Exception as exc:

            self.metrics.increment(
                "errors"
            )

            raise PipelineError(
                f"Unable to flush session "
                f"{session_id}: {exc}"
            ) from exc

    # =================================================================
    # LIST SESSIONS
    # =================================================================

    def list_sessions(self) -> List[Any]:

        manager = self.session_manager

        if hasattr(
            manager,
            "list_sessions",
        ):

            result = manager.list_sessions()

            if result is None:
                return []

            if isinstance(
                result,
                dict,
            ):

                return list(
                    result.values()
                )

            return list(result)

        if hasattr(
            manager,
            "get_sessions",
        ):

            result = manager.get_sessions()

            if result is None:
                return []

            if isinstance(
                result,
                dict,
            ):

                return list(
                    result.values()
                )

            return list(result)

        sessions = getattr(
            manager,
            "sessions",
            None,
        )

        if isinstance(
            sessions,
            dict,
        ):

            return list(
                sessions.values()
            )

        if isinstance(
            sessions,
            (list, tuple),
        ):

            return list(sessions)

        return []

    # =================================================================
    # COUNT
    # =================================================================

    def session_count(self) -> int:

        return len(
            self.list_sessions()
        )

    # =================================================================
    # EVENTS
    # =================================================================

    def process_with_events(
        self,
        session_id: str,
        audio: bytes,
    ) -> Dict[str, Any]:

        result = self.process_audio(
            session_id=session_id,
            audio=audio,
        )

        return result.to_dict()

    # =================================================================
    # HEALTH
    # =================================================================

    def health(self) -> Dict[str, Any]:

        try:

            pipeline_health = (
                self.pipeline.health()
                if hasattr(
                    self.pipeline,
                    "health",
                )
                else {
                    "status": "healthy"
                }
            )

            session_health = (
                self.session_manager.health()
                if hasattr(
                    self.session_manager,
                    "health",
                )
                else {
                    "status": "healthy"
                }
            )

            statuses = (
                pipeline_health,
                session_health,
            )

            if self.state == EngineState.ERROR:

                status = "unhealthy"

            elif any(
                isinstance(item, dict)
                and item.get("status")
                == "unhealthy"
                for item in statuses
            ):

                status = "unhealthy"

            elif any(
                isinstance(item, dict)
                and item.get("status")
                == "degraded"
                for item in statuses
            ):

                status = "degraded"

            elif self.state in (
                EngineState.CREATED,
                EngineState.STOPPED,
            ):

                status = "degraded"

            else:

                status = "healthy"

            return {
                "service":
                    "speakeridentity_diarization_engine",
                "status": status,
                "state": self.state,
                "started_at":
                    self._started_at,
                "stopped_at":
                    self._stopped_at,
                "session_count":
                    self.session_count(),
                "components": {
                    "pipeline":
                        pipeline_health,
                    "session_manager":
                        session_health,
                },
            }

        except Exception as exc:

            return {
                "service":
                    "speakeridentity_diarization_engine",
                "status": "unhealthy",
                "state": self.state,
                "error": str(exc),
            }

    # =================================================================
    # STATISTICS
    # =================================================================

    def statistics(self) -> Dict[str, Any]:

        pipeline_statistics = (
            self.pipeline.statistics()
            if hasattr(
                self.pipeline,
                "statistics",
            )
            else {}
        )

        session_statistics = (
            self.session_manager.statistics()
            if hasattr(
                self.session_manager,
                "statistics",
            )
            else {}
        )

        return {
            "service":
                "speakeridentity_diarization_engine",
            "state":
                self.state,
            "sessions":
                self.session_count(),
            "metrics":
                pipeline_statistics,
            "session_manager":
                session_statistics,
        }

    # =================================================================
    # CONFIGURATION
    # =================================================================

    def configuration(self) -> Dict[str, Any]:

        return self.config.to_dict()

    # =================================================================
    # INTERNAL STATE
    # =================================================================

    def _ensure_running(self) -> None:

        if self.state == EngineState.CREATED:

            self.start()

        if self.state != EngineState.RUNNING:

            raise PipelineError(
                "DiarizationEngine is not running."
            )

    # =================================================================
    # SESSION OPERATION
    # =================================================================

    def _call_session_operation(
        self,
        session_id: str,
        method_names: tuple[str, ...],
    ) -> Any:

        try:

            for method_name in method_names:

                method = getattr(
                    self.pipeline,
                    method_name,
                    None,
                )

                if method is not None:

                    return method(
                        session_id
                    )

            for method_name in method_names:

                method = getattr(
                    self.session_manager,
                    method_name,
                    None,
                )

                if method is not None:

                    return method(
                        session_id
                    )

            raise SessionError(
                "No compatible session operation "
                "is available."
            )

        except (
            SessionNotFoundError,
            SessionError,
            DiarizationError,
        ):

            raise

        except Exception as exc:

            self.metrics.increment(
                "errors"
            )

            raise SessionError(
                f"Unable to execute session operation "
                f"for {session_id}: {exc}"
            ) from exc

    # =================================================================
    # SESSION ID
    # =================================================================

    @staticmethod
    def _validate_session_id(
        session_id: str,
    ) -> str:

        if not isinstance(
            session_id,
            str,
        ):

            raise SessionError(
                "session_id must be a string."
            )

        session_id = session_id.strip()

        if not session_id:

            raise SessionError(
                "session_id cannot be empty."
            )

        if len(session_id) > 128:

            raise SessionError(
                "session_id is too long."
            )

        return session_id

    # =================================================================
    # STOP ALL
    # =================================================================

    def _stop_all_sessions(self) -> None:

        sessions = self.list_sessions()

        for session in sessions:

            if isinstance(
                session,
                dict,
            ):

                session_id = session.get(
                    "session_id"
                )

            else:

                session_id = getattr(
                    session,
                    "session_id",
                    None,
                )

            if not session_id:
                continue

            try:

                self.pipeline.flush(
                    session_id
                )

            except Exception:

                self.metrics.increment(
                    "errors"
                )

            try:

                method = getattr(
                    self.pipeline,
                    "stop_session",
                    None,
                )

                if method is not None:

                    method(
                        session_id
                    )

                else:

                    method = getattr(
                        self.session_manager,
                        "stop_session",
                        None,
                    )

                    if method is not None:

                        method(
                            session_id
                        )

            except Exception:

                self.metrics.increment(
                    "errors"
                )


# ====================================================================
# FACTORY
# ====================================================================

def create_diarization_engine(
    config: Optional[
        DiarizationConfig
    ] = None,
) -> DiarizationEngine:

    return DiarizationEngine(
        config=(
            config
            if config is not None
            else default_config
        )
    )


# ====================================================================
# PUBLIC API
# ====================================================================

__all__ = [
    "EngineState",
    "DiarizationEngine",
    "create_diarization_engine",
]