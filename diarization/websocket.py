"""
====================================================================
SpeakerIdentity IA - Diarization WebSocket
====================================================================

Projet       : SpeakerIdentity IA
Module       : Diarization
Fichier      : diarization/websocket.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 2.0.1
Date         : 14/08/2026

Description
------------

Gestion WebSocket temps réel du module de diarisation SpeakerIdentity IA.

Format audio interne :

    PCM16
    16000 Hz
    Mono
    2 bytes/sample
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time

from dataclasses import dataclass
from typing import Any, Optional

from fastapi import WebSocket, WebSocketDisconnect

from .config import config

from .events import (
    DiarizationEvent,
    EventTypes,
)

from .exceptions import (
    AudioFormatError,
    DiarizationError,
    SessionError,
)

from .metrics import metrics


logger = logging.getLogger(
    "speakeridentity.diarization.websocket"
)


# ====================================================================
# CONNECTION STATE
# ====================================================================

@dataclass
class WebSocketSession:

    session_id: str

    connected_at: float

    audio_chunks: int = 0

    audio_bytes: int = 0

    messages_received: int = 0

    messages_sent: int = 0

    closed: bool = False

    close_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:

        return {
            "session_id": self.session_id,
            "connected_at": self.connected_at,
            "audio_chunks": self.audio_chunks,
            "audio_bytes": self.audio_bytes,
            "messages_received": self.messages_received,
            "messages_sent": self.messages_sent,
            "closed": self.closed,
            "close_reason": self.close_reason,
        }


# ====================================================================
# WEBSOCKET MANAGER
# ====================================================================

class DiarizationWebSocket:

    def __init__(
        self,
        engine: Any,
        *,
        max_message_bytes: Optional[int] = None,
    ) -> None:

        if engine is None:
            raise ValueError(
                "DiarizationEngine is required."
            )

        self.engine = engine

        configured_max_message_bytes = (
            config.websocket_max_message_bytes
            if max_message_bytes is None
            else max_message_bytes
        )

        try:
            self.max_message_bytes = int(
                configured_max_message_bytes
            )
        except (TypeError, ValueError) as exc:

            raise ValueError(
                "max_message_bytes must be an integer."
            ) from exc

        if self.max_message_bytes <= 0:
            raise ValueError(
                "max_message_bytes must be greater than zero."
            )

        self._sessions: dict[
            str,
            WebSocketSession,
        ] = {}

        self._connections: dict[
            str,
            WebSocket,
        ] = {}

        self._lock = asyncio.Lock()

    # =================================================================
    # REGISTER
    # =================================================================

    async def register(
        self,
        session_id: str,
        websocket: WebSocket,
    ) -> WebSocketSession:

        session_id = self._validate_session_id(
            session_id
        )

        if websocket is None:
            raise ValueError(
                "websocket is required."
            )

        async with self._lock:

            if session_id in self._connections:

                raise SessionError(
                    f"Session '{session_id}' "
                    "already has an active WebSocket."
                )

            state = WebSocketSession(
                session_id=session_id,
                connected_at=time.time(),
            )

            self._sessions[session_id] = state

            self._connections[session_id] = websocket

        logger.info(
            "WebSocket registered: session=%s",
            session_id,
        )

        return state

    # =================================================================
    # UNREGISTER
    # =================================================================

    async def unregister(
        self,
        session_id: str,
        *,
        reason: Optional[str] = None,
    ) -> None:

        session_id = self._validate_session_id(
            session_id
        )

        async with self._lock:

            state = self._sessions.pop(
                session_id,
                None,
            )

            self._connections.pop(
                session_id,
                None,
            )

            if state is not None:

                state.closed = True

                state.close_reason = reason

        logger.info(
            "WebSocket unregistered: session=%s reason=%s",
            session_id,
            reason,
        )

    # =================================================================
    # HANDLE
    # =================================================================

    async def handle(
        self,
        websocket: WebSocket,
        session_id: str,
    ) -> None:

        session_id = self._validate_session_id(
            session_id
        )

        if websocket is None:
            raise ValueError(
                "websocket is required."
            )

        state: Optional[
            WebSocketSession
        ] = None

        accepted = False

        try:

            await websocket.accept()

            accepted = True

            state = await self.register(
                session_id,
                websocket,
            )

            await self._ensure_engine_session(
                session_id
            )

            await self._send_event(
                websocket,
                DiarizationEvent(
                    event_type=(
                        EventTypes.SESSION_STARTED
                    ),
                    session_id=session_id,
                    data={
                        "message":
                            "Diarization WebSocket "
                            "session started."
                    },
                ),
                state,
            )

            await self._receive_loop(
                websocket,
                state,
            )

        except WebSocketDisconnect:

            logger.info(
                "WebSocket disconnected: session=%s",
                session_id,
            )

        except asyncio.CancelledError:

            logger.info(
                "WebSocket task cancelled: session=%s",
                session_id,
            )

            raise

        except Exception as exc:

            logger.exception(
                "WebSocket error: session=%s",
                session_id,
            )

            metrics.increment(
                "errors"
            )

            if accepted:

                try:

                    await self._send_error(
                        websocket,
                        session_id,
                        exc,
                        state,
                    )

                except Exception:

                    logger.debug(
                        "Unable to send WebSocket error.",
                        exc_info=True,
                    )

        finally:

            await self._cleanup_session(
                session_id,
                state,
            )

    # =================================================================
    # RECEIVE LOOP
    # =================================================================

    async def _receive_loop(
        self,
        websocket: WebSocket,
        state: WebSocketSession,
    ) -> None:

        while True:

            message = await websocket.receive()

            state.messages_received += 1

            message_type = message.get(
                "type"
            )

            if message_type == (
                "websocket.disconnect"
            ):

                raise WebSocketDisconnect()

            audio = message.get(
                "bytes"
            )

            if audio is not None:

                await self._process_audio(
                    websocket,
                    state,
                    audio,
                )

                continue

            text = message.get(
                "text"
            )

            if text is not None:

                await self._process_text_message(
                    websocket,
                    state,
                    text,
                )

                continue

    # =================================================================
    # AUDIO
    # =================================================================

    async def _process_audio(
        self,
        websocket: WebSocket,
        state: WebSocketSession,
        audio: bytes,
    ) -> None:

        if not isinstance(
            audio,
            bytes,
        ):

            raise AudioFormatError(
                "WebSocket audio payload "
                "must be bytes."
            )

        if not audio:
            return

        if len(audio) > self.max_message_bytes:

            raise AudioFormatError(
                "Audio WebSocket message exceeds "
                "maximum configured size."
            )

        if (
            len(audio)
            % config.sample_width
            != 0
        ):

            raise AudioFormatError(
                "PCM16 audio payload has an "
                "invalid byte length."
            )

        state.audio_chunks += 1

        state.audio_bytes += len(audio)

        metrics.increment(
            "audio_chunks"
        )

        metrics.increment(
            "audio_bytes",
            len(audio),
        )

        logger.debug(
            "Audio received: session=%s "
            "chunk=%d bytes=%d",
            state.session_id,
            state.audio_chunks,
            len(audio),
        )

        await self._send_event(
            websocket,
            DiarizationEvent(
                event_type=(
                    EventTypes.AUDIO_RECEIVED
                ),
                session_id=state.session_id,
                data={
                    "bytes": len(audio),
                    "chunk": state.audio_chunks,
                },
            ),
            state,
        )

        result = await self._process_engine_audio(
            state.session_id,
            audio,
        )

        if result is None:
            return

        await self._send_result(
            websocket,
            state,
            result,
        )

    # =================================================================
    # TEXT COMMAND
    # =================================================================

    async def _process_text_message(
        self,
        websocket: WebSocket,
        state: WebSocketSession,
        text: str,
    ) -> None:

        if not isinstance(
            text,
            str,
        ):

            raise DiarizationError(
                "WebSocket text message "
                "must be a string."
            )

        command = text.strip().lower()

        if command == "ping":

            await self._send_json(
                websocket,
                {
                    "type": "pong",
                    "session_id": state.session_id,
                    "timestamp": time.time(),
                },
                state,
            )

            return

        # -------------------------------------------------------------
        # STATUS
        # -------------------------------------------------------------

        if command == "status":

            status = await self._engine_status(
                state.session_id
            )

            await self._send_json(
                websocket,
                {
                    "type": "session.status",
                    "session_id": state.session_id,
                    "data": status,
                },
                state,
            )

            return

        # -------------------------------------------------------------
        # FLUSH
        # -------------------------------------------------------------

        if command == "flush":

            result = await self._flush_engine_session(
                state.session_id
            )

            if result is not None:

                await self._send_result(
                    websocket,
                    state,
                    result,
                )

            return

        # -------------------------------------------------------------
        # PAUSE
        # -------------------------------------------------------------

        if command == "pause":

            await self._call_engine_method(
                "pause_session",
                state.session_id,
            )

            await self._send_json(
                websocket,
                {
                    "type": "session.paused",
                    "session_id": state.session_id,
                },
                state,
            )

            return

        # -------------------------------------------------------------
        # RESUME
        # -------------------------------------------------------------

        if command == "resume":

            await self._call_engine_method(
                "resume_session",
                state.session_id,
            )

            await self._send_json(
                websocket,
                {
                    "type": "session.resumed",
                    "session_id": state.session_id,
                },
                state,
            )

            return

        # -------------------------------------------------------------
        # STOP
        # -------------------------------------------------------------

        if command == "stop":

            result = await self._call_engine_method(
                "stop_session",
                state.session_id,
            )

            if result is not None:

                await self._send_result(
                    websocket,
                    state,
                    result,
                )

            await self._send_event(
                websocket,
                DiarizationEvent(
                    event_type=(
                        EventTypes.SESSION_STOPPED
                    ),
                    session_id=state.session_id,
                ),
                state,
            )

            try:

                await websocket.close()

            except Exception:

                logger.debug(
                    "WebSocket already closed: "
                    "session=%s",
                    state.session_id,
                    exc_info=True,
                )

            return

        raise DiarizationError(
            f"Unknown WebSocket command: {text}"
        )

    # =================================================================
    # ENGINE AUDIO
    # =================================================================

    async def _process_engine_audio(
        self,
        session_id: str,
        audio: bytes,
    ) -> Any:

        method = getattr(
            self.engine,
            "process_audio",
            None,
        )

        if method is None:

            method = getattr(
                self.engine,
                "process",
                None,
            )

        if method is None:

            raise DiarizationError(
                "DiarizationEngine must expose "
                "'process_audio' or 'process'."
            )

        result = method(
            session_id,
            audio,
        )

        if inspect.isawaitable(result):

            result = await result

        return result

    # =================================================================
    # ENGINE SESSION
    # =================================================================

    async def _ensure_engine_session(
        self,
        session_id: str,
    ) -> None:

        method = getattr(
            self.engine,
            "create_session",
            None,
        )

        if method is None:
            return

        try:

            result = method(
                session_id
            )

            if inspect.isawaitable(result):

                await result

        except SessionError:

            # Une session déjà existante n'est pas
            # considérée comme une erreur ici.
            try:

                existing = self.engine.get_session(
                    session_id
                )

                if existing is not None:
                    return

            except Exception:
                pass

            raise

    # =================================================================
    # ENGINE STATUS
    # =================================================================

    async def _engine_status(
        self,
        session_id: str,
    ) -> dict[str, Any]:

        method = getattr(
            self.engine,
            "get_session",
            None,
        )

        if method is None:

            return {
                "session_id": session_id,
                "status": "unknown",
            }

        result = method(
            session_id
        )

        if inspect.isawaitable(result):

            result = await result

        if hasattr(
            result,
            "to_dict",
        ):

            result = result.to_dict()

        if isinstance(
            result,
            dict,
        ):

            return result

        return {
            "session_id": session_id,
            "status": str(result),
        }

    # =================================================================
    # ENGINE METHOD
    # =================================================================

    async def _call_engine_method(
        self,
        method_name: str,
        session_id: str,
    ) -> Any:

        method = getattr(
            self.engine,
            method_name,
            None,
        )

        if method is None:

            raise DiarizationError(
                f"DiarizationEngine does not expose "
                f"'{method_name}'."
            )

        result = method(
            session_id
        )

        if inspect.isawaitable(result):

            result = await result

        return result

    # =================================================================
    # FLUSH ENGINE
    # =================================================================

    async def _flush_engine_session(
        self,
        session_id: str,
    ) -> Any:

        method = getattr(
            self.engine,
            "flush",
            None,
        )

        if method is None:

            raise DiarizationError(
                "DiarizationEngine does not expose "
                "'flush'."
            )

        result = method(
            session_id
        )

        if inspect.isawaitable(result):

            result = await result

        return result

    # =================================================================
    # SEND RESULT
    # =================================================================

    async def _send_result(
        self,
        websocket: WebSocket,
        state: WebSocketSession,
        result: Any,
    ) -> None:

        if result is None:
            return

        if hasattr(
            result,
            "to_dict",
        ):

            result = result.to_dict()

        if isinstance(
            result,
            dict,
        ):

            payload = {
                "type": EventTypes.DIARIZATION_RESULT,
                "session_id": state.session_id,
                "data": result,
            }

        else:

            payload = {
                "type": EventTypes.DIARIZATION_RESULT,
                "session_id": state.session_id,
                "data": {
                    "result": result,
                },
            }

        await self._send_json(
            websocket,
            payload,
            state,
        )

    # =================================================================
    # SEND EVENT
    # =================================================================

    async def _send_event(
        self,
        websocket: WebSocket,
        event: DiarizationEvent,
        state: WebSocketSession,
    ) -> None:

        if event is None:
            return

        await self._send_json(
            websocket,
            {
                "type": event.event_type,
                "session_id": event.session_id,
                "timestamp": event.timestamp,
                "data": dict(event.data),
            },
            state,
        )

    # =================================================================
    # SEND JSON
    # =================================================================

    async def _send_json(
        self,
        websocket: WebSocket,
        payload: dict[str, Any],
        state: WebSocketSession,
    ) -> None:

        if state.closed:
            return

        await websocket.send_json(
            payload
        )

        state.messages_sent += 1

    # =================================================================
    # ERROR
    # =================================================================

    async def _send_error(
        self,
        websocket: WebSocket,
        session_id: str,
        exc: Exception,
        state: Optional[
            WebSocketSession
        ],
    ) -> None:

        payload = {
            "type": EventTypes.ERROR,
            "session_id": session_id,
            "timestamp": time.time(),
            "data": {
                "error": type(exc).__name__,
                "message": str(exc),
            },
        }

        try:

            await websocket.send_json(
                payload
            )

            if state is not None:
                state.messages_sent += 1

        except Exception:

            logger.debug(
                "Unable to send WebSocket error.",
                exc_info=True,
            )

    # =================================================================
    # CLEANUP
    # =================================================================

    async def _cleanup_session(
        self,
        session_id: str,
        state: Optional[
            WebSocketSession
        ],
    ) -> None:

        # -------------------------------------------------------------
        # IMPORTANT
        # -------------------------------------------------------------
        # Le flush est tenté avant unregister afin de ne pas perdre
        # un segment vocal encore ouvert lors d'une déconnexion.
        # -------------------------------------------------------------

        if state is not None and not state.closed:

            try:

                result = await self._flush_engine_session(
                    session_id
                )

                if result is not None:

                    try:

                        await self._send_result(
                            self._connections.get(
                                session_id
                            ),
                            state,
                            result,
                        )

                    except Exception:

                        logger.debug(
                            "Unable to send final flush result: "
                            "session=%s",
                            session_id,
                            exc_info=True,
                        )

            except Exception:

                logger.debug(
                    "Final session flush failed: "
                    "session=%s",
                    session_id,
                    exc_info=True,
                )

        try:

            await self.unregister(
                session_id,
                reason=(
                    state.close_reason
                    if state is not None
                    and state.close_reason
                    else "connection_closed"
                ),
            )

        except Exception:

            logger.exception(
                "WebSocket unregister failed: "
                "session=%s",
                session_id,
            )

    # =================================================================
    # VALIDATE SESSION ID
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
    # HEALTH
    # =================================================================

    def health(self) -> dict[str, Any]:

        return {
            "status": "healthy",
            "active_connections": len(
                self._connections
            ),
            "max_message_bytes": (
                self.max_message_bytes
            ),
            "sessions": list(
                self._sessions.keys()
            ),
        }

    # =================================================================
    # STATISTICS
    # =================================================================

    def statistics(self) -> dict[str, Any]:

        return {
            "active_connections": len(
                self._connections
            ),
            "sessions": {
                session_id: state.to_dict()
                for session_id, state
                in self._sessions.items()
            },
        }


# ====================================================================
# FACTORY
# ====================================================================

def create_websocket_handler(
    engine: Any,
) -> DiarizationWebSocket:

    return DiarizationWebSocket(
        engine
    )


# ====================================================================
# FASTAPI ENDPOINT
# ====================================================================

def create_websocket_endpoint(
    engine: Any,
):

    handler = create_websocket_handler(
        engine
    )

    async def endpoint(
        websocket: WebSocket,
        session_id: str,
    ) -> None:

        await handler.handle(
            websocket,
            session_id,
        )

    endpoint.handler = handler

    return endpoint


# ====================================================================
# PUBLIC API
# ====================================================================

__all__ = [
    "WebSocketSession",
    "DiarizationWebSocket",
    "create_websocket_handler",
    "create_websocket_endpoint",
]