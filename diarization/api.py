"""
====================================================================
SpeakerIdentity IA - Diarization API
====================================================================

Projet       : SpeakerIdentity IA
Module       : Diarization
Fichier      : diarization/api.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 2.0.0
Date         : 14/08/2026

Description
------------

API REST et WebSocket du module de diarisation SpeakerIdentity IA.

Responsabilités :

    - exposition REST du module ;
    - health check ;
    - statistiques ;
    - gestion des sessions ;
    - consultation d'une session ;
    - suppression/arrêt d'une session ;
    - exposition WebSocket ;
    - gestion standardisée des erreurs.

Architecture :

    FastAPI
       |
       +-------------------+
       |                   |
       v                   v
     REST              WebSocket
       |                   |
       +---------+---------+
                 |
                 v
       DiarizationEngine
                 |
                 v
       SessionManager
                 |
                 v
       Diarization Pipeline

Important
---------

Ce fichier ne crée aucun modèle lourd.

Il ne doit pas :

    - charger Whisper ;
    - charger un modèle d'embedding ;
    - initialiser CUDA ;
    - créer plusieurs EmbeddingEngine.

Les composants sont injectés depuis l'application principale.

Endpoints :

    GET
        /api/diarization/health

    GET
        /api/diarization/statistics

    GET
        /api/diarization/sessions

    GET
        /api/diarization/sessions/{session_id}

    POST
        /api/diarization/sessions/{session_id}

    POST
        /api/diarization/sessions/{session_id}/start

    POST
        /api/diarization/sessions/{session_id}/pause

    POST
        /api/diarization/sessions/{session_id}/resume

    POST
        /api/diarization/sessions/{session_id}/stop

    DELETE
        /api/diarization/sessions/{session_id}

    WebSocket
        /ws/diarization/{session_id}

====================================================================
"""

from __future__ import annotations

import inspect
import logging

from typing import Any, Optional


# ====================================================================
# FASTAPI
# ====================================================================

from fastapi import (
    APIRouter,
    HTTPException,
    WebSocket,
)


# ====================================================================
# LOCAL IMPORTS
# ====================================================================

from .config import config

from .exceptions import (
    DiarizationError,
    SessionError,
    SessionNotFoundError,
)

from .health import (
    DiarizationHealth,
)

from .metrics import (
    metrics,
)

from .websocket import (
    DiarizationWebSocket,
    create_websocket_handler,
)


# ====================================================================
# LOGGER
# ====================================================================

logger = logging.getLogger(
    "speakeridentity.diarization.api"
)


# ====================================================================
# API CONTEXT
# ====================================================================


class DiarizationAPI:
    """
    Contexte API du module de diarisation.

    Les composants lourds sont injectés.

    Exemple :

        api = DiarizationAPI(
            engine=engine,
            session_manager=session_manager,
        )
    """

    def __init__(
        self,
        engine: Any,
        session_manager: Any = None,
        *,
        health: Optional[
            DiarizationHealth
        ] = None,
    ) -> None:

        if engine is None:
            raise ValueError(
                "DiarizationEngine is required."
            )

        self.engine = engine

        # -------------------------------------------------------------
        # SESSION MANAGER
        # -------------------------------------------------------------

        if session_manager is None:

            session_manager = getattr(
                engine,
                "session_manager",
                None,
            )

        self.session_manager = (
            session_manager
        )

        # -------------------------------------------------------------
        # HEALTH
        # -------------------------------------------------------------

        if health is None:

            health = DiarizationHealth()

            health.register(
                "engine",
                engine,
            )

            if (
                session_manager is not None
            ):
                health.register(
                    "session_manager",
                    session_manager,
                )

        self.health_manager = health

        # -------------------------------------------------------------
        # WEBSOCKET
        # -------------------------------------------------------------

        self.websocket_handler = (
            create_websocket_handler(
                engine
            )
        )

    # =================================================================
    # HEALTH
    # =================================================================

    async def health_check(
        self,
    ) -> dict[str, Any]:
        """
        Vérifie l'état global du module.
        """

        result = (
            self.health_manager.check()
        )

        # -------------------------------------------------------------
        # METRICS
        # -------------------------------------------------------------

        result["metrics"] = (
            metrics.snapshot_dict()
        )

        return result

    # =================================================================
    # STATISTICS
    # =================================================================

    async def statistics(
        self,
    ) -> dict[str, Any]:
        """
        Retourne les statistiques du module.
        """

        result = {
            "service": "speakeridentity_diarization",
            "metrics": (
                metrics.snapshot_dict()
            ),
        }

        # -------------------------------------------------------------
        # ENGINE STATISTICS
        # -------------------------------------------------------------

        engine_stats = await self._call_optional(
            self.engine,
            "statistics",
        )

        if engine_stats is not None:

            result["engine"] = (
                self._serialize(
                    engine_stats
                )
            )

        # -------------------------------------------------------------
        # SESSION STATISTICS
        # -------------------------------------------------------------

        if (
            self.session_manager
            is not None
        ):

            session_stats = (
                await self._call_optional(
                    self.session_manager,
                    "statistics",
                )
            )

            if session_stats is not None:

                result["sessions"] = (
                    self._serialize(
                        session_stats
                    )
                )

        # -------------------------------------------------------------
        # WEBSOCKET STATISTICS
        # -------------------------------------------------------------

        result["websocket"] = (
            self.websocket_handler.statistics()
        )

        return result

    # =================================================================
    # LIST SESSIONS
    # =================================================================

    async def list_sessions(
        self,
    ) -> dict[str, Any]:
        """
        Retourne la liste des sessions actives.
        """

        if (
            self.session_manager
            is None
        ):

            return {
                "sessions": [],
                "count": 0,
            }

        result = await self._call_optional(
            self.session_manager,
            "list_sessions",
        )

        if result is None:

            result = await self._call_optional(
                self.session_manager,
                "get_sessions",
            )

        if result is None:

            return {
                "sessions": [],
                "count": 0,
            }

        result = self._serialize(
            result
        )

        if isinstance(
            result,
            dict,
        ):

            if "sessions" in result:

                sessions = result[
                    "sessions"
                ]

            else:

                sessions = []

        elif isinstance(
            result,
            list,
        ):

            sessions = result

        else:

            sessions = []

        return {
            "sessions": sessions,
            "count": len(sessions),
        }

    # =================================================================
    # GET SESSION
    # =================================================================

    async def get_session(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Retourne une session.
        """

        session_id = (
            self._validate_session_id(
                session_id
            )
        )

        # -------------------------------------------------------------
        # ENGINE FIRST
        # -------------------------------------------------------------

        result = await self._call_optional(
            self.engine,
            "get_session",
            session_id,
        )

        # -------------------------------------------------------------
        # SESSION MANAGER
        # -------------------------------------------------------------

        if result is None:

            if (
                self.session_manager
                is not None
            ):

                result = (
                    await self._call_optional(
                        self.session_manager,
                        "get_session",
                        session_id,
                    )
                )

        if result is None:

            raise SessionNotFoundError(
                f"Session '{session_id}' "
                "not found."
            )

        result = self._serialize(
            result
        )

        if isinstance(
            result,
            dict,
        ):
            return result

        return {
            "session_id": session_id,
            "data": result,
        }

    # =================================================================
    # CREATE SESSION
    # =================================================================

    async def create_session(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Crée une session de diarisation.
        """

        session_id = (
            self._validate_session_id(
                session_id
            )
        )

        method = getattr(
            self.engine,
            "create_session",
            None,
        )

        if method is None:

            raise SessionError(
                "DiarizationEngine does not "
                "provide create_session()."
            )

        result = method(
            session_id
        )

        if inspect.isawaitable(
            result
        ):
            result = await result

        result = self._serialize(
            result
        )

        return {
            "success": True,
            "session_id": session_id,
            "session": result,
        }

    # =================================================================
    # START SESSION
    # =================================================================

    async def start_session(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Démarre une session.
        """

        return await self._session_action(
            session_id,
            (
                "start_session",
                "start",
            ),
            "started",
        )

    # =================================================================
    # PAUSE SESSION
    # =================================================================

    async def pause_session(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Met une session en pause.
        """

        return await self._session_action(
            session_id,
            (
                "pause_session",
                "pause",
            ),
            "paused",
        )

    # =================================================================
    # RESUME SESSION
    # =================================================================

    async def resume_session(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Reprend une session.
        """

        return await self._session_action(
            session_id,
            (
                "resume_session",
                "resume",
            ),
            "resumed",
        )

    # =================================================================
    # STOP SESSION
    # =================================================================

    async def stop_session(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Arrête une session.
        """

        return await self._session_action(
            session_id,
            (
                "stop_session",
                "stop",
            ),
            "stopped",
        )

    # =================================================================
    # DELETE SESSION
    # =================================================================

    async def delete_session(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Supprime une session.
        """

        session_id = (
            self._validate_session_id(
                session_id
            )
        )

        method = getattr(
            self.engine,
            "delete_session",
            None,
        )

        if method is None:

            method = getattr(
                self.session_manager,
                "delete_session",
                None,
            )

        if method is None:

            raise SessionError(
                "No delete_session() method "
                "available."
            )

        result = method(
            session_id
        )

        if inspect.isawaitable(
            result
        ):
            result = await result

        return {
            "success": True,
            "session_id": session_id,
            "status": "deleted",
            "result": self._serialize(
                result
            ),
        }

    # =================================================================
    # WEBSOCKET
    # =================================================================

    async def websocket(
        self,
        websocket: WebSocket,
        session_id: str,
    ) -> None:
        """
        Point d'entrée WebSocket.
        """

        session_id = (
            self._validate_session_id(
                session_id
            )
        )

        await self.websocket_handler.handle(
            websocket,
            session_id,
        )

    # =================================================================
    # SESSION ACTION
    # =================================================================

    async def _session_action(
        self,
        session_id: str,
        method_names: tuple[str, ...],
        status: str,
    ) -> dict[str, Any]:
        """
        Exécute une action sur une session.
        """

        session_id = (
            self._validate_session_id(
                session_id
            )
        )

        method = None

        # -------------------------------------------------------------
        # ENGINE
        # -------------------------------------------------------------

        for method_name in method_names:

            method = getattr(
                self.engine,
                method_name,
                None,
            )

            if method is not None:
                break

        # -------------------------------------------------------------
        # SESSION MANAGER
        # -------------------------------------------------------------

        if method is None:

            if (
                self.session_manager
                is not None
            ):

                for method_name in (
                    method_names
                ):

                    method = getattr(
                        self.session_manager,
                        method_name,
                        None,
                    )

                    if method is not None:
                        break

        # -------------------------------------------------------------
        # NOT FOUND
        # -------------------------------------------------------------

        if method is None:

            raise SessionError(
                f"No method available for "
                f"session action '{status}'."
            )

        # -------------------------------------------------------------
        # CALL
        # -------------------------------------------------------------

        result = method(
            session_id
        )

        if inspect.isawaitable(
            result
        ):
            result = await result

        return {
            "success": True,
            "session_id": session_id,
            "status": status,
            "result": self._serialize(
                result
            ),
        }

    # =================================================================
    # OPTIONAL CALL
    # =================================================================

    @staticmethod
    async def _call_optional(
        component: Any,
        method_name: str,
        *args: Any,
    ) -> Any:
        """
        Appelle une méthode si elle existe.
        """

        if component is None:
            return None

        method = getattr(
            component,
            method_name,
            None,
        )

        if method is None:
            return None

        result = method(
            *args
        )

        if inspect.isawaitable(
            result
        ):
            result = await result

        return result

    # =================================================================
    # SERIALIZATION
    # =================================================================

    @staticmethod
    def _serialize(
        value: Any,
    ) -> Any:
        """
        Convertit les objets SpeakerIdentity en structures JSON.
        """

        if value is None:
            return None

        if hasattr(
            value,
            "to_dict",
        ):

            return value.to_dict()

        if isinstance(
            value,
            list,
        ):

            return [
                DiarizationAPI._serialize(
                    item
                )
                for item in value
            ]

        if isinstance(
            value,
            tuple,
        ):

            return [
                DiarizationAPI._serialize(
                    item
                )
                for item in value
            ]

        if isinstance(
            value,
            dict,
        ):

            return {
                key: DiarizationAPI._serialize(
                    item
                )
                for key, item
                in value.items()
            }

        return value

    # =================================================================
    # VALIDATION
    # =================================================================

    @staticmethod
    def _validate_session_id(
        session_id: str,
    ) -> str:
        """
        Valide un identifiant de session.
        """

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


# ====================================================================
# ROUTER FACTORY
# ====================================================================


def create_diarization_router(
    engine: Any,
    session_manager: Any = None,
    *,
    prefix: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> APIRouter:
    """
    Crée le routeur FastAPI du module.

    Exemple :

        router = create_diarization_router(
            engine=diarization_engine,
            session_manager=session_manager,
        )

        app.include_router(router)
    """

    if not config.api_enabled:

        raise RuntimeError(
            "Diarization API is disabled "
            "by configuration."
        )

    api = DiarizationAPI(
        engine=engine,
        session_manager=session_manager,
    )

    router = APIRouter(
        prefix=(
            config.api_prefix
            if prefix is None
            else prefix
        ),
        tags=(
            tags
            if tags is not None
            else ["SpeakerIdentity Diarization"]
        ),
    )

    # ================================================================
    # HEALTH
    # ================================================================

    @router.get(
        config.health_endpoint,
    )
    async def health() -> dict[str, Any]:

        return await api.health_check()

    # ================================================================
    # STATISTICS
    # ================================================================

    @router.get(
        config.stats_endpoint,
    )
    async def statistics() -> dict[str, Any]:

        return await api.statistics()

    # ================================================================
    # LIST SESSIONS
    # ================================================================

    @router.get(
        "/sessions",
    )
    async def list_sessions() -> dict[str, Any]:

        return await api.list_sessions()

    # ================================================================
    # GET SESSION
    # ================================================================

    @router.get(
        "/sessions/{session_id}",
    )
    async def get_session(
        session_id: str,
    ) -> dict[str, Any]:

        try:

            return await api.get_session(
                session_id
            )

        except SessionNotFoundError as exc:

            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc

    # ================================================================
    # CREATE SESSION
    # ================================================================

    @router.post(
        "/sessions/{session_id}",
    )
    async def create_session(
        session_id: str,
    ) -> dict[str, Any]:

        try:

            return await api.create_session(
                session_id
            )

        except SessionError as exc:

            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc

    # ================================================================
    # START
    # ================================================================

    @router.post(
        "/sessions/{session_id}/start",
    )
    async def start_session(
        session_id: str,
    ) -> dict[str, Any]:

        try:

            return await api.start_session(
                session_id
            )

        except SessionNotFoundError as exc:

            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc

        except SessionError as exc:

            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc

    # ================================================================
    # PAUSE
    # ================================================================

    @router.post(
        "/sessions/{session_id}/pause",
    )
    async def pause_session(
        session_id: str,
    ) -> dict[str, Any]:

        try:

            return await api.pause_session(
                session_id
            )

        except SessionError as exc:

            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc

    # ================================================================
    # RESUME
    # ================================================================

    @router.post(
        "/sessions/{session_id}/resume",
    )
    async def resume_session(
        session_id: str,
    ) -> dict[str, Any]:

        try:

            return await api.resume_session(
                session_id
            )

        except SessionError as exc:

            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc

    # ================================================================
    # STOP
    # ================================================================

    @router.post(
        "/sessions/{session_id}/stop",
    )
    async def stop_session(
        session_id: str,
    ) -> dict[str, Any]:

        try:

            return await api.stop_session(
                session_id
            )

        except SessionError as exc:

            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc

    # ================================================================
    # DELETE
    # ================================================================

    @router.delete(
        "/sessions/{session_id}",
    )
    async def delete_session(
        session_id: str,
    ) -> dict[str, Any]:

        try:

            return await api.delete_session(
                session_id
            )

        except SessionNotFoundError as exc:

            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc

        except SessionError as exc:

            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc

    # ================================================================
    # WEBSOCKET
    # ================================================================

    if config.websocket_enabled:

        @router.websocket(
            config.websocket_path
            + "/{session_id}"
        )
        async def diarization_websocket(
            websocket: WebSocket,
            session_id: str,
        ) -> None:

            await api.websocket(
                websocket,
                session_id,
            )

    return router


# ====================================================================
# COMPATIBILITY ALIAS
# ====================================================================


create_router = create_diarization_router


# ====================================================================
# PUBLIC API
# ====================================================================


__all__ = [
    "DiarizationAPI",
    "create_diarization_router",
    "create_router",
]