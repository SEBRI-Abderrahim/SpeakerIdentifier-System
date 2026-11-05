"""
====================================================================
SpeakerIdentity IA - Diarization Events
====================================================================

Projet       : SpeakerIdentity IA
Module       : Diarization
Fichier      : diarization/events.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 2.1.1
Date         : 14/08/2026

Description
------------

Gestion centralisée des événements du module de diarisation SpeakerIdentity IA.

Ce module définit le contrat d'événements utilisé par :

    - AudioProcessor
    - VAD
    - SegmentManager
    - SpeakerManager
    - SessionManager
    - Pipeline
    - DiarizationEngine
    - WebSocket
    - API
    - Monitoring
    - intégration future STT
    - Frontend SpeakerIdentity

Principe
--------

Tous les événements transportent au minimum :

    - un type ;
    - un session_id ;
    - un timestamp.

Les informations complémentaires sont placées dans
le champ ``data``.

IMPORTANT
---------

Ce module ne calcule pas la confidence speaker.

La confidence doit être fournie par le composant responsable
de l'identification du speaker.

Exemple :

    speaker_manager
        |
        +--> speaker_id
        +--> confidence
        |
        v
    speaker_identified_event()
        |
        v
    data["speaker_id"]
    data["confidence"]

Aucune confidence par défaut n'est injectée ici.

Multi-session
-------------

Chaque événement est obligatoirement associé à une session.

Exemple :

    session_A -> speaker_01
    session_B -> speaker_01

Les deux événements restent complètement indépendants
grâce au ``session_id``.

====================================================================
"""

from __future__ import annotations

import time
import uuid

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from .models import DiarizationEventType


# ====================================================================
# EVENT NAMES
# ====================================================================


class EventTypes:
    """
    Compatibilité et accès simplifié aux types d'événements.

    Les valeurs sont directement basées sur
    ``DiarizationEventType`` défini dans ``models.py``.

    ``DiarizationEventType`` reste la source de vérité unique.
    """

    SESSION_STARTED = DiarizationEventType.SESSION_STARTED.value
    SESSION_STOPPED = DiarizationEventType.SESSION_STOPPED.value

    AUDIO_RECEIVED = DiarizationEventType.AUDIO_RECEIVED.value

    SPEECH_STARTED = DiarizationEventType.SPEECH_STARTED.value
    SPEECH_ENDED = DiarizationEventType.SPEECH_ENDED.value

    SEGMENT_CREATED = DiarizationEventType.SEGMENT_CREATED.value

    SPEAKER_DETECTED = DiarizationEventType.SPEAKER_DETECTED.value
    SPEAKER_IDENTIFIED = DiarizationEventType.SPEAKER_IDENTIFIED.value
    SPEAKER_CREATED = DiarizationEventType.SPEAKER_CREATED.value

    DIARIZATION_RESULT = (
        DiarizationEventType.DIARIZATION_RESULT.value
    )

    ERROR = DiarizationEventType.ERROR.value

    @classmethod
    def all(cls) -> tuple[str, ...]:
        """
        Retourne tous les types d'événements applicatifs.
        """

        return tuple(
            event_type.value
            for event_type in DiarizationEventType
        )


# ====================================================================
# EVENT TYPE CONVERSION
# ====================================================================


def normalize_event_type(
    event_type: str | DiarizationEventType,
) -> str:
    """
    Normalise un type d'événement.

    Accepte :

        - DiarizationEventType
        - str

    Retourne toujours une chaîne.
    """

    if isinstance(
        event_type,
        DiarizationEventType,
    ):
        value = event_type.value

    elif isinstance(
        event_type,
        str,
    ):
        value = event_type.strip()

    else:
        raise TypeError(
            "event_type must be a string or "
            "DiarizationEventType."
        )

    if not value:
        raise ValueError(
            "event_type must not be empty."
        )

    return value


# ====================================================================
# DIARIZATION EVENT
# ====================================================================


@dataclass(slots=True)
class DiarizationEvent:
    """
    Événement standard produit par le système SpeakerIdentity.

    IMPORTANT
    ---------

    Cette classe ne définit aucune confidence par défaut.

    Exemple :

        DiarizationEvent(
            event_type=EventTypes.SPEAKER_IDENTIFIED,
            session_id="session_001",
            data={
                "speaker_id": "speaker_01",
                "confidence": 0.91,
            },
        )

    La valeur ``confidence`` est entièrement déterminée
    par le composant producteur.

    ``events.py`` se limite à transporter cette information.
    """

    event_type: str | DiarizationEventType

    session_id: str

    timestamp: float = field(
        default_factory=time.time
    )

    data: Dict[str, Any] = field(
        default_factory=dict
    )

    event_id: str = field(
        default_factory=lambda: (
            f"event_{uuid.uuid4().hex[:12]}"
        )
    )

    # =================================================================
    # VALIDATION
    # =================================================================

    def __post_init__(self) -> None:
        """
        Valide et normalise l'événement.

        Aucune valeur métier telle que la confidence
        n'est créée ou modifiée ici.
        """

        # -------------------------------------------------------------
        # EVENT TYPE
        # -------------------------------------------------------------

        self.event_type = normalize_event_type(
            self.event_type
        )

        # -------------------------------------------------------------
        # SESSION ID
        # -------------------------------------------------------------

        if self.session_id is None:
            raise ValueError(
                "session_id must not be empty."
            )

        if not isinstance(
            self.session_id,
            str,
        ):
            raise TypeError(
                "session_id must be a string."
            )

        self.session_id = self.session_id.strip()

        if not self.session_id:
            raise ValueError(
                "session_id must not be empty."
            )

        # -------------------------------------------------------------
        # TIMESTAMP
        # -------------------------------------------------------------

        try:
            self.timestamp = float(
                self.timestamp
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "timestamp must be a valid number."
            ) from exc

        if self.timestamp < 0:
            raise ValueError(
                "timestamp must be >= 0."
            )

        # -------------------------------------------------------------
        # DATA
        # -------------------------------------------------------------

        if self.data is None:

            self.data = {}

        elif not isinstance(
            self.data,
            Mapping,
        ):

            raise ValueError(
                "data must be a mapping."
            )

        else:

            # Copie indépendante.
            #
            # IMPORTANT :
            # aucune valeur n'est ajoutée ici.
            self.data = dict(
                self.data
            )

        # -------------------------------------------------------------
        # EVENT ID
        # -------------------------------------------------------------

        if self.event_id is None:

            self.event_id = (
                f"event_{uuid.uuid4().hex[:12]}"
            )

        elif not isinstance(
            self.event_id,
            str,
        ):

            raise TypeError(
                "event_id must be a string."
            )

        else:

            self.event_id = self.event_id.strip()

        if not self.event_id:

            self.event_id = (
                f"event_{uuid.uuid4().hex[:12]}"
            )

    # =================================================================
    # FACTORY
    # =================================================================

    @classmethod
    def create(
        cls,
        event_type: str | DiarizationEventType,
        session_id: str,
        *,
        data: Optional[
            Mapping[str, Any]
        ] = None,
        timestamp: Optional[float] = None,
    ) -> "DiarizationEvent":
        """
        Factory permettant de créer un événement.

        IMPORTANT
        ---------

        ``data`` est transmis tel quel après copie.

        Aucun champ ``confidence`` n'est ajouté
        automatiquement.
        """

        return cls(
            event_type=event_type,
            session_id=session_id,
            timestamp=(
                time.time()
                if timestamp is None
                else timestamp
            ),
            data=dict(
                data
                if data is not None
                else {}
            ),
        )

    # =================================================================
    # SERIALIZATION
    # =================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit l'événement en dictionnaire.

        Le contenu de ``data`` est conservé tel qu'il a
        été fourni par le producteur.

        Aucun champ confidence n'est ajouté.
        """

        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "data": dict(self.data),
        }

    # =================================================================
    # JSON-FRIENDLY
    # =================================================================

    def as_payload(self) -> Dict[str, Any]:
        """
        Retourne le payload réseau.

        Cette méthode utilise le même contrat que ``to_dict``.
        """

        return self.to_dict()

    # =================================================================
    # HELPERS
    # =================================================================

    @property
    def type(self) -> str:
        """
        Alias court pour ``event_type``.
        """

        return self.event_type

    def is_type(
        self,
        event_type: str | DiarizationEventType,
    ) -> bool:
        """
        Vérifie le type de l'événement.
        """

        return (
            self.event_type
            == normalize_event_type(
                event_type
            )
        )

    def copy(
        self,
        *,
        data: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> "DiarizationEvent":
        """
        Crée une copie indépendante de l'événement.

        Un nouvel event_id est généré.

        Si ``data`` n'est pas fourni, les données originales
        sont conservées sans modification.
        """

        return DiarizationEvent(
            event_type=self.event_type,
            session_id=self.session_id,
            timestamp=self.timestamp,
            data=dict(
                self.data
                if data is None
                else data
            ),
        )


# ====================================================================
# EVENT FACTORIES
# ====================================================================


def session_started_event(
    session_id: str,
    **data: Any,
) -> DiarizationEvent:
    """
    Crée un événement de démarrage de session.
    """

    return DiarizationEvent.create(
        EventTypes.SESSION_STARTED,
        session_id,
        data=data,
    )


def session_stopped_event(
    session_id: str,
    **data: Any,
) -> DiarizationEvent:
    """
    Crée un événement d'arrêt de session.
    """

    return DiarizationEvent.create(
        EventTypes.SESSION_STOPPED,
        session_id,
        data=data,
    )


def audio_received_event(
    session_id: str,
    **data: Any,
) -> DiarizationEvent:
    """
    Crée un événement de réception audio.
    """

    return DiarizationEvent.create(
        EventTypes.AUDIO_RECEIVED,
        session_id,
        data=data,
    )


def speech_started_event(
    session_id: str,
    **data: Any,
) -> DiarizationEvent:
    """
    Crée un événement de début de parole.
    """

    return DiarizationEvent.create(
        EventTypes.SPEECH_STARTED,
        session_id,
        data=data,
    )


def speech_ended_event(
    session_id: str,
    **data: Any,
) -> DiarizationEvent:
    """
    Crée un événement de fin de parole.
    """

    return DiarizationEvent.create(
        EventTypes.SPEECH_ENDED,
        session_id,
        data=data,
    )


def segment_created_event(
    session_id: str,
    **data: Any,
) -> DiarizationEvent:
    """
    Crée un événement de création de segment.
    """

    return DiarizationEvent.create(
        EventTypes.SEGMENT_CREATED,
        session_id,
        data=data,
    )


def speaker_detected_event(
    session_id: str,
    **data: Any,
) -> DiarizationEvent:
    """
    Crée un événement de détection de speaker.

    La confidence, si elle existe, doit être fournie
    par le producteur :

        speaker_detected_event(
            session_id,
            speaker_id="speaker_01",
            confidence=calculated_confidence,
        )

    Aucune confidence n'est ajoutée automatiquement.
    """

    return DiarizationEvent.create(
        EventTypes.SPEAKER_DETECTED,
        session_id,
        data=data,
    )


def speaker_identified_event(
    session_id: str,
    **data: Any,
) -> DiarizationEvent:
    """
    Crée un événement d'identification de speaker.

    La confidence doit provenir du composant
    d'identification.

    Exemple :

        speaker_identified_event(
            session_id,
            speaker_id=speaker_id,
            confidence=confidence,
        )

    Aucune valeur par défaut n'est appliquée.
    """

    return DiarizationEvent.create(
        EventTypes.SPEAKER_IDENTIFIED,
        session_id,
        data=data,
    )


def speaker_created_event(
    session_id: str,
    **data: Any,
) -> DiarizationEvent:
    """
    Crée un événement de création de speaker.

    La confidence éventuelle est transmise uniquement
    si le producteur la fournit.
    """

    return DiarizationEvent.create(
        EventTypes.SPEAKER_CREATED,
        session_id,
        data=data,
    )


def diarization_result_event(
    session_id: str,
    **data: Any,
) -> DiarizationEvent:
    """
    Crée un événement contenant un résultat
    de diarisation.

    Les segments, speakers et confidences présents
    dans ``data`` sont transmis sans modification.
    """

    return DiarizationEvent.create(
        EventTypes.DIARIZATION_RESULT,
        session_id,
        data=data,
    )


def error_event(
    session_id: str,
    *,
    error: str,
    error_type: Optional[str] = None,
    **data: Any,
) -> DiarizationEvent:
    """
    Crée un événement d'erreur.

    ``error`` contient le message utilisateur/API.

    ``error_type`` permet de conserver le type
    d'exception technique.
    """

    payload: Dict[str, Any] = {
        "error": str(error),
        **data,
    }

    if error_type is not None:

        payload["error_type"] = str(
            error_type
        )

    return DiarizationEvent.create(
        EventTypes.ERROR,
        session_id,
        data=payload,
    )


# ====================================================================
# PUBLIC API
# ====================================================================

__all__ = [
    # Event contract
    "DiarizationEvent",
    "DiarizationEventType",

    # Event types
    "EventTypes",

    # Helpers
    "normalize_event_type",

    # Factories
    "session_started_event",
    "session_stopped_event",
    "audio_received_event",
    "speech_started_event",
    "speech_ended_event",
    "segment_created_event",
    "speaker_detected_event",
    "speaker_identified_event",
    "speaker_created_event",
    "diarization_result_event",
    "error_event",
]