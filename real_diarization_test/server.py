"""
====================================================================
SpeakerIdentity IA - Real Diarization Test Server
====================================================================

Projet       : SpeakerIdentity IA
Module       : Real Diarization Test
Fichier      : real_diarization_test/server.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 1.0.9
Date         : 15/08/2026

Description
------------

Serveur de test réel multi-sessions :

    Microphone navigateur
            |
            v
        WebSocket
            |
            v
        PCM16 16kHz
            |
            v
        DiarizationPipeline
            |
            +--> AudioProcessor
            +--> VAD
            +--> SegmentManager
            +--> EmbeddingEngine partagé
            +--> SpeakerManager
            +--> SessionManager
            |
            v
        Résultats navigateur

GESTION DES SESSIONS
--------------------

Chaque session WebSocket possède sa propre instance de
DiarizationPipeline.

Ainsi :

    Session A -> Pipeline A
    Session B -> Pipeline B
    Session C -> Pipeline C

Les états de traitement audio, VAD, segmentation et
SpeakerManager ne sont donc pas mélangés entre les sessions.

IMPORTANT
---------

Le modèle Speaker Embedding est partagé globalement.

Une seule instance de EmbeddingEngine est créée pour
toute la durée de vie du processus Python.

Cette même instance est utilisée par :

    - toutes les sessions
    - tous les speakers
    - tous les pipelines

Le modèle SpeechBrain ECAPA est donc chargé une seule fois
par processus serveur.

Le traitement CPU du pipeline est exécuté dans un thread
afin qu'une session lourde ne bloque pas les autres connexions
WebSocket.

Authentification Hugging Face
----------------------------

Le modèle public SpeechBrain est utilisé sans HF_TOKEN.

Aucun token Hugging Face n'est nécessaire pour ce test local.

Les logs techniques Hugging Face, SpeechBrain, HTTPX et
HTTPCore sont masqués.

Les erreurs réelles restent visibles.

Protocole WebSocket :

    Client -> JSON start
    Client -> binary PCM16
    Client -> JSON ping
    Client -> JSON stop

    Server -> session_started
    Server -> diarization_result
    Server -> flush_result
    Server -> error
====================================================================
"""

from __future__ import annotations


# ====================================================================
# IMPORTS
# ====================================================================

import asyncio
import json
import logging
import sys
import uuid

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any


from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import FileResponse


# ====================================================================
# PROJECT ROOT
# ====================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ====================================================================
# LOGGING FILTER
# ====================================================================

class ThirdPartyLogFilter(
    logging.Filter
):
    """
    Filtre les logs techniques des bibliothèques tierces.

    Les messages INFO/WARNING provenant de :

        - Hugging Face
        - SpeechBrain
        - HTTPX
        - HTTPCore

    sont masqués.

    Les erreurs ERROR/CRITICAL restent visibles.
    """

    _PREFIXES = (
        "huggingface_hub",
        "httpx",
        "httpcore",
        "speechbrain",
    )

    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:

        logger_name = record.name

        is_third_party = any(
            logger_name == prefix
            or logger_name.startswith(
                prefix + "."
            )
            for prefix in self._PREFIXES
        )

        if not is_third_party:
            return True

        return (
            record.levelno
            >= logging.ERROR
        )


# ====================================================================
# LOGGING CONFIGURATION
# ====================================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)


# ====================================================================
# CONFIGURATION DES LOGGERS TIERS
# ====================================================================

THIRD_PARTY_LOGGERS = (
    "huggingface_hub",
    "huggingface_hub.utils",
    "huggingface_hub.utils._http",
    "huggingface_hub.utils._validators",
    "huggingface_hub.utils.fetching",
    "huggingface_hub.file_download",
    "httpx",
    "httpcore",
    "speechbrain",
    "speechbrain.utils",
    "speechbrain.utils.fetching",
    "speechbrain.utils.parameter_transfer",
    "speechbrain.utils.data_utils",
    "speechbrain.dataio",
    "speechbrain.processing",
)


for logger_name in THIRD_PARTY_LOGGERS:

    third_party_logger = logging.getLogger(
        logger_name
    )

    third_party_logger.setLevel(
        logging.ERROR
    )

    third_party_logger.propagate = True


# ====================================================================
# APPLICATION DU FILTRE
# ====================================================================

def _install_log_filters() -> None:
    """
    Installe le filtre sur les handlers existants.
    """

    root_logger = logging.getLogger()

    log_filter = ThirdPartyLogFilter()

    for handler in root_logger.handlers:

        handler.addFilter(
            log_filter
        )

    for logger_name in THIRD_PARTY_LOGGERS:

        third_party_logger = logging.getLogger(
            logger_name
        )

        for handler in (
            third_party_logger.handlers
        ):

            handler.addFilter(
                log_filter
            )


_install_log_filters()


# ====================================================================
# LOGGER SpeakerIdentity
# ====================================================================

logger = logging.getLogger(
    "speakeridentity.real_diarization_test"
)


# ====================================================================
# SpeakerIdentity IMPORTS
# ====================================================================

from diarization.config import (
    config as diarization_config,
)

from diarization.embedding_engine import (
    EmbeddingEngine,
)

from diarization.pipeline import (
    DiarizationPipeline,
)

from diarization.session_manager import (
    DiarizationSessionManager,
)


# ====================================================================
# RÉINSTALLATION DES FILTRES
# ====================================================================

_install_log_filters()


# ====================================================================
# SHARED EMBEDDING ENGINE
# ====================================================================

shared_embedding_engine = EmbeddingEngine(
    config=diarization_config,
    auto_load=False,
)


# ====================================================================
# SESSION MANAGER
# ====================================================================

session_manager = (
    DiarizationSessionManager(
        config=diarization_config
    )
)


# ====================================================================
# SESSION PIPELINES
# ====================================================================

session_pipelines: dict[
    str,
    DiarizationPipeline,
] = {}


# ====================================================================
# CLIENT CONNECTIONS
# ====================================================================

connections: dict[
    str,
    WebSocket,
] = {}


# ====================================================================
# SESSION RESULT STATE
# ====================================================================

session_results: dict[
    str,
    dict[str, list[Any]],
] = {}


# ====================================================================
# APPLICATION LIFESPAN
# ====================================================================

@asynccontextmanager
async def lifespan(
    application: FastAPI,
):
    """
    Cycle de vie de l'application FastAPI.

    Le modèle ECAPA est chargé une seule fois.
    """

    logger.info(
        "Initializing shared speaker embedding model..."
    )

    try:

        if shared_embedding_engine.is_loaded:

            logger.info(
                "Shared speaker embedding model "
                "already loaded. Reusing existing instance."
            )

        else:

            shared_embedding_engine.load()

            if not shared_embedding_engine.is_loaded:

                raise RuntimeError(
                    "Speaker embedding model failed to load."
                )

            logger.info(
                "Shared speaker embedding model "
                "loaded successfully."
            )

        logger.info(
            "Shared embedding model ready for all "
            "sessions and pipelines."
        )

        yield

    except Exception as exc:

        logger.error("")
        logger.error("=" * 68)
        logger.error(
            "  MODELE D'EMPREINTE VOCALE INDISPONIBLE"
        )
        logger.error("=" * 68)
        logger.error("")
        logger.error(
            "  %s : %s",
            type(exc).__name__,
            exc,
        )
        logger.error("")

        missing = []

        for module, package in (
            ("torch", "torch"),
            ("speechbrain", "speechbrain"),
        ):

            try:

                __import__(module)

            except ImportError:

                missing.append(package)

        if missing:

            logger.error(
                "  Paquet(s) absent(s) : %s",
                ", ".join(missing),
            )

            logger.error("")

            logger.error(
                "      pip install %s",
                " ".join(missing),
            )

        else:

            logger.error(
                "  torch et speechbrain sont installes."
            )

            logger.error(
                "  Le modele n'a pas pu etre charge."
            )

            logger.error("")

            logger.error(
                "  Verifier la connexion Internet et "
                "le cache Hugging Face."
            )

        logger.error("")
        logger.error("=" * 68)
        logger.error("")

        raise

    finally:

        logger.info(
            "SpeakerIdentity real diarization application shutdown."
        )


# ====================================================================
# FASTAPI APPLICATION
# ====================================================================

app = FastAPI(
    title="SpeakerIdentity IA - Real Diarization Test",
    version="1.0.9",
    lifespan=lifespan,
)


# ====================================================================
# CORS
# ====================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================================================================
# HOME
# ====================================================================

@app.get("/")
async def index():
    """
    Retourne le client Web.
    """

    index_file = (
        Path(__file__).parent
        / "index.html"
    )

    if not index_file.exists():

        logger.error(
            "index.html not found: %s",
            index_file,
        )

        return {
            "status": "error",
            "error": (
                "real_diarization_test/index.html "
                "not found."
            ),
        }

    return FileResponse(
        index_file
    )


# ====================================================================
# HEALTH PAGE
# ====================================================================

@app.get("/health.html")
async def health_page():
    """
    Retourne l'interface Web Health.

    IMPORTANT :

    L'endpoint /health reste strictement réservé
    à l'API JSON.
    """

    health_file = (
        Path(__file__).parent
        / "health.html"
    )

    if not health_file.exists():

        logger.error(
            "health.html not found: %s",
            health_file,
        )

        return {
            "status": "error",
            "error": (
                "real_diarization_test/health.html "
                "not found."
            ),
        }

    return FileResponse(
        health_file
    )


# ====================================================================
# STATISTICS PAGE
# ====================================================================

@app.get("/statistics.html")
async def statistics_page():
    """
    Retourne l'interface Web Statistics.

    IMPORTANT :

    L'endpoint /statistics reste strictement réservé
    à l'API JSON.
    """

    statistics_file = (
        Path(__file__).parent
        / "statistics.html"
    )

    if not statistics_file.exists():

        logger.error(
            "statistics.html not found: %s",
            statistics_file,
        )

        return {
            "status": "error",
            "error": (
                "real_diarization_test/statistics.html "
                "not found."
            ),
        }

    return FileResponse(
        statistics_file
    )


# ====================================================================
# DIARIZATION PAGE
# ====================================================================

@app.get("/diarization.html")
async def diarization_page():
    """
    Retourne l'interface Web Diarization.

    Le WebSocket reste strictement :

        /ws/diarization
    """

    diarization_file = (
        Path(__file__).parent
        / "diarization.html"
    )

    if not diarization_file.exists():

        logger.error(
            "diarization.html not found: %s",
            diarization_file,
        )

        return {
            "status": "error",
            "error": (
                "real_diarization_test/diarization.html "
                "not found."
            ),
        }

    return FileResponse(
        diarization_file
    )


# ====================================================================
# THRESHOLD GET
# ====================================================================

@app.get("/threshold")
async def lire_seuil():
    """
    Retourne le seuil d'identification actuel.
    """

    return {
        "threshold": float(
            diarization_config
            .speaker_similarity_threshold
        ),
        "aide": (
            "POST /threshold?valeur=0.40 "
            "pour modifier le seuil."
        ),
    }


# ====================================================================
# THRESHOLD POST
# ====================================================================

@app.post("/threshold")
async def poser_seuil(
    valeur: float,
):
    """
    Modifie le seuil d'identification.
    """

    if not (
        0.0
        <= valeur
        <= 1.0
    ):

        return {
            "erreur": (
                "Le seuil doit être compris "
                "entre 0.0 et 1.0."
            ),
            "recu": valeur,
        }

    ancien = float(
        diarization_config
        .speaker_similarity_threshold
    )

    diarization_config.speaker_similarity_threshold = (
        float(valeur)
    )

    logger.info(
        "Seuil d'identification : %.3f -> %.3f",
        ancien,
        valeur,
    )

    return {
        "ancien": ancien,
        "threshold": float(valeur),
    }


# ====================================================================
# HEALTH
# ====================================================================

@app.get("/health")
async def health():
    """
    Health check du serveur.
    """

    try:

        embedding_health = (
            shared_embedding_engine.health()
        )

        session_health = (
            session_manager.health()
        )

        embedding_status = (
            embedding_health.get("status")
            if isinstance(
                embedding_health,
                dict,
            )
            else None
        )

        session_status = (
            session_health.get("status")
            if isinstance(
                session_health,
                dict,
            )
            else None
        )

        overall_status = "healthy"

        if (
            embedding_status
            not in (
                None,
                "healthy",
            )
            or session_status
            not in (
                None,
                "healthy",
            )
        ):

            overall_status = "degraded"

        return {
            "status": overall_status,
            "service": (
                "speakeridentity_real_diarization_test"
            ),
            "sessions": {
                "active": len(
                    session_pipelines
                ),
                "connections": len(
                    connections
                ),
            },
            "embedding_engine": embedding_health,
            "session_manager": session_health,
        }

    except Exception as exc:

        logger.exception(
            "Health check failed."
        )

        return {
            "status": "degraded",
            "service": (
                "speakeridentity_real_diarization_test"
            ),
            "sessions": {
                "active": len(
                    session_pipelines
                ),
                "connections": len(
                    connections
                ),
            },
            "error": str(exc),
        }


# ====================================================================
# STATISTICS
# ====================================================================

@app.get("/statistics")
async def statistics():
    """
    Retourne les statistiques.
    """

    try:

        return {
            "embedding_engine": (
                shared_embedding_engine.statistics()
            ),
            "sessions": (
                session_manager.statistics()
            ),
            "active_sessions": len(
                session_pipelines
            ),
            "active_connections": len(
                connections
            ),
        }

    except Exception as exc:

        logger.exception(
            "Statistics request failed."
        )

        return {
            "error": str(exc),
        }


# ====================================================================
# CREATE SESSION PIPELINE
# ====================================================================

def _create_session_pipeline() -> DiarizationPipeline:
    """
    Crée un pipeline indépendant pour une session.

    IMPORTANT :

    Le pipeline possède ses propres composants d'état :

        - AudioProcessor
        - VAD
        - SegmentManager
        - SpeakerManager

    mais réutilise :

        - le même SessionManager ;
        - le même EmbeddingEngine.

    Cela permet l'exécution de plusieurs sessions simultanées
    tout en conservant un seul modèle ECAPA en mémoire.
    """

    return DiarizationPipeline(
        session_manager=session_manager,
        config=diarization_config,
        embedding_engine=shared_embedding_engine,
    )


# ====================================================================
# STORE RESULT
# ====================================================================

def _store_pipeline_result(
    session_id: str,
    payload: dict[str, Any],
) -> None:
    """
    Conserve les segments et événements effectivement produits
    par le pipeline pour la session.

    Un résultat vide n'est pas considéré comme une erreur.
    """

    state = session_results.setdefault(
        session_id,
        {
            "segments": [],
            "events": [],
        },
    )

    segments = payload.get(
        "segments",
        [],
    )

    events = payload.get(
        "events",
        [],
    )

    if segments:

        state["segments"].extend(
            segments
        )

    if events:

        state["events"].extend(
            events
        )


# ====================================================================
# WEBSOCKET
# ====================================================================

@app.websocket(
    "/ws/diarization"
)
async def diarization_websocket(
    websocket: WebSocket,
):
    """
    WebSocket réel microphone -> diarisation.

    Plusieurs connexions peuvent fonctionner simultanément.

    Protocole :

        Client -> JSON start
        Client -> binary PCM16
        Client -> JSON ping
        Client -> JSON stop

        Server -> session_started
        Server -> diarization_result
        Server -> flush_result
        Server -> error
    """

    await websocket.accept()

    connection_id = str(
        uuid.uuid4()
    )

    connections[
        connection_id
    ] = websocket

    session_id: str | None = None

    disconnected = False

    logger.info(
        "WebSocket connected: %s",
        connection_id,
    )

    try:

        # ============================================================
        # WAIT START
        # ============================================================

        message = await websocket.receive()

        message_type = message.get(
            "type"
        )

        if (
            message_type
            == "websocket.disconnect"
        ):

            disconnected = True

            return

        text = message.get(
            "text"
        )

        if not text:

            await safe_send_json(
                websocket,
                {
                    "type": "error",
                    "error": (
                        "Expected start JSON message."
                    ),
                },
            )

            return

        # ============================================================
        # PARSE START
        # ============================================================

        try:

            data = json.loads(
                text
            )

        except json.JSONDecodeError as exc:

            logger.warning(
                "Invalid start JSON from %s: %s",
                connection_id,
                exc,
            )

            await safe_send_json(
                websocket,
                {
                    "type": "error",
                    "error": (
                        "Invalid JSON start message."
                    ),
                },
            )

            return

        if data.get(
            "action"
        ) != "start":

            await safe_send_json(
                websocket,
                {
                    "type": "error",
                    "error": (
                        "First message must be "
                        "'start'."
                    ),
                },
            )

            return

        # ============================================================
        # SESSION PARAMETERS
        # ============================================================

        requested_session_id = data.get(
            "session_id"
        )

        metadata = data.get(
            "metadata",
            {},
        )

        if metadata is None:

            metadata = {}

        if not isinstance(
            metadata,
            dict,
        ):

            await safe_send_json(
                websocket,
                {
                    "type": "error",
                    "error": (
                        "metadata must be an object."
                    ),
                },
            )

            return

        # ============================================================
        # CREATE INDEPENDENT PIPELINE
        # ============================================================

        session_pipeline = (
            _create_session_pipeline()
        )

        # ============================================================
        # START SESSION
        # ============================================================

        try:

            session = await asyncio.to_thread(
                session_pipeline.start_session,
                session_id=(
                    requested_session_id
                    if requested_session_id
                    else None
                ),
                metadata=metadata,
            )

        except Exception:

            logger.exception(
                "Unable to start session "
                "connection=%s",
                connection_id,
            )

            await safe_send_json(
                websocket,
                {
                    "type": "error",
                    "error": (
                        "Unable to start diarization session."
                    ),
                },
            )

            return

        session_id = session.session_id

        # ============================================================
        # REGISTER SESSION PIPELINE
        # ============================================================

        session_pipelines[
            session_id
        ] = session_pipeline

        session_results[
            session_id
        ] = {
            "segments": [],
            "events": [],
        }

        session_status = (
            session.status.value
            if hasattr(
                session.status,
                "value",
            )
            else str(
                session.status
            )
        )

        logger.info(
            "Session started: %s | "
            "connection=%s | "
            "active_sessions=%d",
            session_id,
            connection_id,
            len(session_pipelines),
        )

        # ============================================================
        # SESSION STARTED
        # ============================================================

        sent = await safe_send_json(
            websocket,
            {
                "type": "session_started",
                "session_id": session_id,
                "audio": {
                    "codec": "PCM16",
                    "sample_rate": (
                        diarization_config.sample_rate
                    ),
                    "channels": (
                        diarization_config.channels
                    ),
                    "sample_width": (
                        diarization_config.sample_width
                    ),
                },
                "status": session_status,
                "embedding": {
                    "shared": True,
                    "loaded": (
                        shared_embedding_engine.is_loaded
                    ),
                    "model": (
                        shared_embedding_engine.model_name
                    ),
                    "device": (
                        shared_embedding_engine.device
                    ),
                },
            },
        )

        if not sent:

            disconnected = True

            return

        # ============================================================
        # AUDIO LOOP
        # ============================================================

        while True:

            message = await websocket.receive()

            message_type = message.get(
                "type"
            )

            # ========================================================
            # DISCONNECT
            # ========================================================

            if (
                message_type
                == "websocket.disconnect"
            ):

                disconnected = True

                logger.info(
                    "WebSocket disconnected "
                    "during session=%s",
                    session_id,
                )

                break

            # ========================================================
            # TEXT
            # ========================================================

            text = message.get(
                "text"
            )

            if text is not None:

                try:

                    data = json.loads(
                        text
                    )

                except json.JSONDecodeError:

                    await safe_send_json(
                        websocket,
                        {
                            "type": "error",
                            "session_id": session_id,
                            "error": (
                                "Invalid JSON."
                            ),
                        },
                    )

                    continue

                action = data.get(
                    "action"
                )

                # ====================================================
                # STOP
                # ====================================================

                if action == "stop":

                    logger.info(
                        "Stop requested "
                        "for session=%s",
                        session_id,
                    )

                    break

                # ====================================================
                # PING
                # ====================================================

                if action == "ping":

                    await safe_send_json(
                        websocket,
                        {
                            "type": "pong",
                            "session_id": session_id,
                        },
                    )

                    continue

                continue

            # ========================================================
            # BINARY AUDIO
            # ========================================================

            audio = message.get(
                "bytes"
            )

            if audio is None:
                continue

            if not audio:
                continue

            # ========================================================
            # PIPELINE VALIDATION
            # ========================================================

            if session_id is None:

                await safe_send_json(
                    websocket,
                    {
                        "type": "error",
                        "error": (
                            "No active session."
                        ),
                    },
                )

                continue

            session_pipeline = (
                session_pipelines.get(
                    session_id
                )
            )

            if session_pipeline is None:

                await safe_send_json(
                    websocket,
                    {
                        "type": "error",
                        "session_id": session_id,
                        "error": (
                            "Session pipeline not found."
                        ),
                    },
                )

                break

            # ========================================================
            # PCM16
            # ========================================================

            sample_width = int(
                diarization_config.sample_width
            )

            if (
                sample_width <= 0
                or len(audio)
                % sample_width
                != 0
            ):

                logger.warning(
                    "Invalid PCM buffer size=%d "
                    "session=%s",
                    len(audio),
                    session_id,
                )

                await safe_send_json(
                    websocket,
                    {
                        "type": "error",
                        "session_id": session_id,
                        "error": (
                            "PCM16 buffer has "
                            "an invalid byte length."
                        ),
                    },
                )

                continue

            # ========================================================
            # PROCESS
            # ========================================================

            try:

                result = await asyncio.to_thread(
                    session_pipeline.process,
                    session_id,
                    audio,
                )

            except Exception as exc:

                logger.exception(
                    "Pipeline processing failed "
                    "for session=%s",
                    session_id,
                )

                await safe_send_json(
                    websocket,
                    {
                        "type": "error",
                        "session_id": session_id,
                        "error": (
                            "Diarization processing failed."
                        ),
                        "details": str(exc),
                    },
                )

                continue

            if result is None:
                continue

            # ========================================================
            # SERIALIZE
            # ========================================================

            try:

                payload = result.to_dict()

            except Exception as exc:

                logger.exception(
                    "Unable to serialize pipeline result "
                    "for session=%s",
                    session_id,
                )

                await safe_send_json(
                    websocket,
                    {
                        "type": "error",
                        "session_id": session_id,
                        "error": (
                            "Invalid diarization result."
                        ),
                        "details": str(exc),
                    },
                )

                continue

            # ========================================================
            # STORE REAL RESULTS
            # ========================================================

            _store_pipeline_result(
                session_id,
                payload,
            )

            segments = payload.get(
                "segments",
                [],
            )

            events = payload.get(
                "events",
                [],
            )

            # ========================================================
            # EMPTY CHUNK
            # ========================================================

            if not segments and not events:

                logger.debug(
                    "Diarization chunk contains no "
                    "completed segment/event: "
                    "session=%s",
                    session_id,
                )

                continue

            # ========================================================
            # SEND REAL RESULT
            # ========================================================

            sent = await safe_send_json(
                websocket,
                {
                    "type": "diarization_result",
                    **payload,
                },
            )

            if not sent:

                disconnected = True

                logger.warning(
                    "Failed to send diarization "
                    "result for session=%s",
                    session_id,
                )

                break

    # =================================================================
    # WEBSOCKET DISCONNECT
    # =================================================================

    except WebSocketDisconnect:

        disconnected = True

        logger.info(
            "WebSocket disconnected: %s",
            connection_id,
        )

    # =================================================================
    # GENERAL ERROR
    # =================================================================

    except Exception as exc:

        logger.exception(
            "Diarization test error "
            "connection=%s session=%s",
            connection_id,
            session_id,
        )

        if not disconnected:

            await safe_send_json(
                websocket,
                {
                    "type": "error",
                    "session_id": session_id,
                    "error": str(exc),
                },
            )

    # =================================================================
    # CLEANUP
    # =================================================================

    finally:

        # ============================================================
        # SESSION PIPELINE
        # ============================================================

        session_pipeline = None

        if session_id:

            session_pipeline = (
                session_pipelines.get(
                    session_id
                )
            )

        # ============================================================
        # FLUSH
        # ============================================================

        if (
            session_id
            and session_pipeline is not None
        ):

            try:

                logger.info(
                    "Flushing session=%s",
                    session_id,
                )

                result = await asyncio.to_thread(
                    session_pipeline.flush,
                    session_id,
                )

                if result is not None:

                    payload = result.to_dict()

                    _store_pipeline_result(
                        session_id,
                        payload,
                    )

                    flush_segments = payload.get(
                        "segments",
                        [],
                    )

                    flush_events = payload.get(
                        "events",
                        [],
                    )

                    # =================================================
                    # SEND FLUSH ONLY IF CONNECTION IS ALIVE
                    # =================================================

                    if not disconnected:

                        await safe_send_json(
                            websocket,
                            {
                                "type": "flush_result",
                                "session_id": session_id,
                                **payload,
                            },
                        )

                    logger.info(
                        "Flush completed: "
                        "session=%s segments=%d events=%d",
                        session_id,
                        len(flush_segments),
                        len(flush_events),
                    )

            except Exception as exc:

                logger.warning(
                    "Flush failed for %s: %s",
                    session_id,
                    exc,
                )

            # ========================================================
            # STOP SESSION
            # ========================================================

            try:

                logger.info(
                    "Stopping session=%s",
                    session_id,
                )

                await asyncio.to_thread(
                    session_pipeline.stop_session,
                    session_id,
                )

            except Exception as exc:

                logger.warning(
                    "Session stop failed "
                    "for %s: %s",
                    session_id,
                    exc,
                )

            # ========================================================
            # REMOVE SESSION PIPELINE
            # ========================================================

            session_pipelines.pop(
                session_id,
                None,
            )

            # ========================================================
            # REMOVE SESSION RESULT STATE
            # ========================================================

            session_results.pop(
                session_id,
                None,
            )

            logger.info(
                "Session resources released: %s | "
                "active_sessions=%d",
                session_id,
                len(session_pipelines),
            )

        # ============================================================
        # REMOVE CONNECTION
        # ============================================================

        connections.pop(
            connection_id,
            None,
        )

        logger.info(
            "Connection closed: %s | "
            "active_connections=%d",
            connection_id,
            len(connections),
        )


# ====================================================================
# SAFE SEND JSON
# ====================================================================

async def safe_send_json(
    websocket: WebSocket,
    payload: dict[str, Any],
) -> bool:
    """
    Envoie un JSON au WebSocket sans faire échouer
    le nettoyage de la connexion.
    """

    try:

        await websocket.send_json(
            payload
        )

        return True

    except Exception as exc:

        logger.debug(
            "WebSocket send failed: %s",
            exc,
        )

        return False


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":

    import uvicorn

    logger.info(
        "Starting SpeakerIdentity real diarization test server..."
    )

    logger.info(
        "Project root: %s",
        PROJECT_ROOT,
    )

    logger.info(
        "Web interface: "
        "http://127.0.0.1:8010/"
    )

    logger.info(
        "Health endpoint: "
        "http://127.0.0.1:8010/health"
    )

    logger.info(
        "WebSocket endpoint: "
        "ws://127.0.0.1:8010/ws/diarization"
    )

    logger.info(
        "Multi-session mode: ENABLED"
    )

    logger.info(
        "Shared embedding engine: ENABLED"
    )

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8010,
        log_level="info",
    )