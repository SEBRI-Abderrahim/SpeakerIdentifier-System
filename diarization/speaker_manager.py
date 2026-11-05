"""
====================================================================
SpeakerIdentity IA - Speaker Manager
====================================================================

Projet       : SpeakerIdentity IA
Module       : Diarization
Fichier      : diarization/speaker_manager.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 2.2.0
Date         : 15/08/2026

Description
------------

Gestionnaire d'identification des speakers du module de
diarisation SpeakerIdentity IA.

Responsabilités :

    - gestion des speakers par session ;
    - comparaison des embeddings ;
    - identification d'un speaker existant ;
    - création d'un nouveau speaker ;
    - calcul de la confiance ;
    - détection des changements de speaker ;
    - isolation stricte des sessions ;
    - support multi-session ;
    - thread-safety ;
    - compatibilité avec SpeakerDiarizer.

IMPORTANT
---------

Le SpeakerManager ne génère pas les embeddings.

Les embeddings sont produits en amont par le moteur
Speaker Embedding.

Pipeline :

    Audio
       |
       v
    VAD
       |
       v
    Speaker Embedding
       |
       v
    SpeakerDiarizer
       |
       v
    SpeakerManager
       |
       v
    speaker_xxxxxxxx

====================================================================
"""

from __future__ import annotations


# ====================================================================
# IMPORTS
# ====================================================================

import logging
import threading
import time
import uuid

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ====================================================================
# LOCAL IMPORTS
# ====================================================================

from .config import (
    DiarizationConfig,
    config as default_config,
)

from .exceptions import (
    DiarizationError,
    SessionError,
)


# ====================================================================
# LOGGER
# ====================================================================

logger = logging.getLogger(
    "speakeridentity.diarization.speaker_manager"
)


# ====================================================================
# CONSTANTES
# ====================================================================

UNKNOWN_SPEAKER_ID = "speaker_unknown"

# --------------------------------------------------------------------
# Seuil volontairement plus robuste que 0.75.
#
# Avec ECAPA-TDNN, une valeur trop élevée peut provoquer la création
# d'un nouveau speaker pour chaque segment.
#
# Le seuil configuré par DiarizationConfig reste prioritaire.
# --------------------------------------------------------------------

DEFAULT_SIMILARITY_THRESHOLD = 0.55

# --------------------------------------------------------------------
# Nombre maximum d'embeddings conservés par speaker.
#
# Cela permet de conserver une représentation robuste de la voix
# au lieu de dépendre d'un seul segment.
# --------------------------------------------------------------------

MAX_EMBEDDING_HISTORY = 10

# --------------------------------------------------------------------
# Nombre de similarites retenues pour decider.
#
# Voir la note dans _speaker_similarity : on prend la moyenne des N
# meilleures et non le maximum, pour qu'un vecteur isole ne puisse pas
# fusionner deux locuteurs a lui seul.
# --------------------------------------------------------------------

_SIMILARITES_RETENUES = 3

# --------------------------------------------------------------------
# Marge d'apprentissage.
#
# Un segment n'est memorise dans l'historique du locuteur que s'il
# depasse le seuil d'au moins cette marge. Les segments limites sont
# reconnus, mais ne servent pas de reference : c'est ce qui empeche la
# derive irreversible constatee lors de l'audit.
# --------------------------------------------------------------------

_MARGE_APPRENTISSAGE = 0.05

# --------------------------------------------------------------------
# Marge d'ambiguite entre deux locuteurs.
#
# Si les deux meilleurs candidats sont separes de moins que cela, le
# segment est attribue mais rien n'est appris : on ne sait pas de qui
# il vient.
# --------------------------------------------------------------------

_MARGE_AMBIGUITE = 0.05


# ====================================================================
# SPEAKER
# ====================================================================

@dataclass
class Speaker:
    """
    Représentation interne d'un speaker.
    """

    speaker_id: str

    session_id: str

    embedding: Any

    confidence: float = 0.0

    segment_count: int = 0

    created_at: float = field(
        default_factory=time.time
    )

    last_seen: float = field(
        default_factory=time.time
    )

    active: bool = True

    # ----------------------------------------------------------------
    # Historique des embeddings du speaker.
    # ----------------------------------------------------------------

    embedding_history: List[np.ndarray] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit le speaker en dictionnaire.
        """

        return {
            "speaker_id": self.speaker_id,
            "session_id": self.session_id,
            "confidence": float(
                self.confidence
            ),
            "segment_count": int(
                self.segment_count
            ),
            "created_at": float(
                self.created_at
            ),
            "last_seen": float(
                self.last_seen
            ),
            "active": bool(
                self.active
            ),
        }


# ====================================================================
# IDENTIFICATION RESULT
# ====================================================================

@dataclass
class SpeakerIdentificationResult:
    """
    Résultat complet d'identification d'un speaker.

    Cette structure est volontairement compatible avec
    SpeakerDiarizer.
    """

    session_id: str

    speaker_id: str

    confidence: float

    created: bool = False

    similarity: float = 0.0

    distance: float = 0.0

    is_new_speaker: bool = False

    is_speaker_change: bool = False

    previous_speaker_id: Optional[str] = None

    timestamp: Optional[float] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    # ----------------------------------------------------------------
    # Compatibilité éventuelle avec anciennes interfaces.
    # ----------------------------------------------------------------

    @property
    def identified(self) -> bool:
        """
        Indique si le speaker a été identifié.
        """

        return (
            self.speaker_id != UNKNOWN_SPEAKER_ID
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Conversion en dictionnaire.
        """

        return {
            "session_id": self.session_id,
            "speaker_id": self.speaker_id,
            "confidence": float(
                max(
                    0.0,
                    min(
                        1.0,
                        self.confidence,
                    ),
                )
            ),
            "created": bool(
                self.created
            ),
            "similarity": float(
                max(
                    -1.0,
                    min(
                        1.0,
                        self.similarity,
                    ),
                )
            ),
            "distance": float(
                max(
                    0.0,
                    self.distance,
                )
            ),
            "is_new_speaker": bool(
                self.is_new_speaker
            ),
            "is_speaker_change": bool(
                self.is_speaker_change
            ),
            "previous_speaker_id": (
                self.previous_speaker_id
            ),
            "timestamp": self.timestamp,
            "identified": self.identified,
            "metadata": dict(
                self.metadata
            ),
        }


# ====================================================================
# SPEAKER MANAGER
# ====================================================================

class SpeakerManager:
    """
    Gestionnaire central des speakers.

    Les speakers sont strictement isolés par session.

    Structure interne :

        {
            session_id: {
                speaker_id: Speaker
            }
        }

    Le manager ne partage jamais un speaker entre deux sessions.
    """

    def __init__(
        self,
        config: Optional[
            DiarizationConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else default_config
        )

        self.config.validate()

        self._lock = threading.RLock()

        self._speakers: Dict[
            str,
            Dict[str, Speaker],
        ] = {}

        # -------------------------------------------------------------
        # Speaker courant par session.
        # -------------------------------------------------------------

        self._current_speaker: Dict[
            str,
            Optional[str],
        ] = {}

        # -------------------------------------------------------------
        # Nombre de segments traités.
        # -------------------------------------------------------------

        self._identified_segments = 0

        self._created_speakers = 0

        logger.debug(
            "SpeakerManager initialized."
        )

    # =================================================================
    # CREATE SESSION
    # =================================================================

    def create_session(
        self,
        session_id: str,
    ) -> None:
        """
        Crée une session si elle n'existe pas.

        Cette méthode est idempotente.
        """

        session_id = self._validate_session_id(
            session_id
        )

        with self._lock:

            self._speakers.setdefault(
                session_id,
                {},
            )

            self._current_speaker.setdefault(
                session_id,
                None,
            )

    # =================================================================
    # HAS SESSION
    # =================================================================

    def has_session(
        self,
        session_id: str,
    ) -> bool:
        """
        Vérifie si une session existe.
        """

        session_id = self._validate_session_id(
            session_id
        )

        with self._lock:

            return session_id in self._speakers

    # =================================================================
    # REMOVE SESSION
    # =================================================================

    def remove_session(
        self,
        session_id: str,
    ) -> bool:
        """
        Supprime une session complète.
        """

        session_id = self._validate_session_id(
            session_id
        )

        with self._lock:

            existed = (
                session_id in self._speakers
            )

            self._speakers.pop(
                session_id,
                None,
            )

            self._current_speaker.pop(
                session_id,
                None,
            )

            return existed

    # =================================================================
    # RESET SESSION
    # =================================================================

    def reset_session(
        self,
        session_id: str,
    ) -> None:
        """
        Réinitialise complètement une session.
        """

        session_id = self._validate_session_id(
            session_id
        )

        with self._lock:

            self._speakers[
                session_id
            ] = {}

            self._current_speaker[
                session_id
            ] = None

    # =================================================================
    # IDENTIFY
    # =================================================================

    def identify(
        self,
        embedding: Any,
        session_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> SpeakerIdentificationResult:
        """
        Identifie un speaker à partir d'un embedding.

        Si un speaker existant est suffisamment proche,
        celui-ci est réutilisé.

        Sinon un nouveau speaker est créé.
        """

        session_id = self._validate_session_id(
            session_id
        )

        normalized_embedding = (
            self._normalize_embedding(
                embedding
            )
        )

        metadata_copy = dict(
            metadata or {}
        )

        with self._lock:

            self.create_session(
                session_id
            )

            speakers = self._speakers[
                session_id
            ]

            previous_speaker_id = (
                self._current_speaker.get(
                    session_id
                )
            )

            # =========================================================
            # PREMIER SPEAKER
            # =========================================================

            if not speakers:

                result = self._create_speaker(
                    session_id=session_id,
                    embedding=normalized_embedding,
                    timestamp=timestamp,
                    metadata=metadata_copy,
                )

                result.previous_speaker_id = (
                    previous_speaker_id
                )

                result.is_speaker_change = (
                    previous_speaker_id is not None
                    and previous_speaker_id
                    != result.speaker_id
                )

                self._current_speaker[
                    session_id
                ] = result.speaker_id

                return result

            # =========================================================
            # COMPARAISON
            # =========================================================

            best_speaker: Optional[
                Speaker
            ] = None

            best_similarity = -2.0

            # CORRECTIF : on suit aussi le deuxieme meilleur candidat.
            # Quand les deux sont trop proches, le segment est attribue
            # mais rien n'est memorise : on ne sait pas de qui il vient,
            # et l'enregistrer polluerait durablement un des deux profils.
            second_similarity = -2.0

            for speaker in speakers.values():

                similarity = (
                    self._speaker_similarity(
                        normalized_embedding,
                        speaker,
                    )
                )

                if similarity > best_similarity:

                    second_similarity = best_similarity

                    best_similarity = similarity

                    best_speaker = speaker

                elif similarity > second_similarity:

                    second_similarity = similarity

            # =========================================================
            # SEUIL
            # =========================================================

            threshold = (
                self._get_similarity_threshold()
            )

            logger.debug(
                "Speaker comparison: "
                "session=%s best=%s "
                "similarity=%.4f threshold=%.4f",
                session_id,
                (
                    best_speaker.speaker_id
                    if best_speaker is not None
                    else None
                ),
                best_similarity,
                threshold,
            )

            # =========================================================
            # MATCH
            # =========================================================

            if (
                best_speaker is not None
                and best_similarity >= threshold
            ):

                confidence = (
                    self._similarity_to_confidence(
                        best_similarity,
                        threshold,
                    )
                )

                # Le segment n'entre dans l'historique du locuteur que
                # s'il depasse nettement le seuil ET qu'aucun autre
                # locuteur n'est aussi proche.
                franc = (
                    best_similarity
                    >= threshold + _MARGE_APPRENTISSAGE
                )

                ambigu = (
                    second_similarity > -1.5
                    and (
                        best_similarity - second_similarity
                    )
                    < _MARGE_AMBIGUITE
                )

                self._update_speaker(
                    speaker=best_speaker,
                    embedding=normalized_embedding,
                    confidence=confidence,
                    timestamp=timestamp,
                    apprendre=franc and not ambigu,
                )

                is_change = (
                    previous_speaker_id is not None
                    and previous_speaker_id
                    != best_speaker.speaker_id
                )

                self._current_speaker[
                    session_id
                ] = best_speaker.speaker_id

                self._identified_segments += 1

                return SpeakerIdentificationResult(
                    session_id=session_id,
                    speaker_id=best_speaker.speaker_id,
                    confidence=confidence,
                    created=False,
                    similarity=best_similarity,
                    distance=max(
                        0.0,
                        1.0 - best_similarity,
                    ),
                    is_new_speaker=False,
                    is_speaker_change=is_change,
                    previous_speaker_id=(
                        previous_speaker_id
                    ),
                    timestamp=timestamp,
                    metadata=metadata_copy,
                )

            # =========================================================
            # NO MATCH
            # =========================================================

            result = self._create_speaker(
                session_id=session_id,
                embedding=normalized_embedding,
                similarity=best_similarity,
                timestamp=timestamp,
                metadata=metadata_copy,
            )

            result.previous_speaker_id = (
                previous_speaker_id
            )

            result.is_speaker_change = (
                previous_speaker_id is not None
                and previous_speaker_id
                != result.speaker_id
            )

            self._current_speaker[
                session_id
            ] = result.speaker_id

            return result

    # =================================================================
    # IDENTIFY OR CREATE
    # =================================================================

    def identify_or_create(
        self,
        embedding: Any,
        session_id: Optional[str] = None,
    ) -> SpeakerIdentificationResult:
        """
        Alias historique.
        """

        return self.identify(
            embedding=embedding,
            session_id=session_id,
        )

    # =================================================================
    # CREATE SPEAKER
    # =================================================================

    def _create_speaker(
        self,
        session_id: str,
        embedding: np.ndarray,
        similarity: float = 0.0,
        timestamp: Optional[float] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> SpeakerIdentificationResult:
        """
        Crée un nouveau speaker.
        """

        speaker_id = (
            self._generate_speaker_id()
        )

        now = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        speaker = Speaker(
            speaker_id=speaker_id,
            session_id=session_id,
            embedding=embedding.copy(),
            confidence=1.0,
            segment_count=1,
            created_at=now,
            last_seen=now,
            active=True,
            embedding_history=[
                embedding.copy()
            ],
        )

        self._speakers.setdefault(
            session_id,
            {},
        )[speaker_id] = speaker

        self._created_speakers += 1

        logger.info(
            "New speaker created: "
            "session=%s speaker=%s",
            session_id,
            speaker_id,
        )

        return SpeakerIdentificationResult(
            session_id=session_id,
            speaker_id=speaker_id,
            confidence=1.0,
            created=True,
            similarity=max(
                -1.0,
                min(
                    1.0,
                    similarity,
                ),
            ),
            distance=max(
                0.0,
                1.0 - max(
                    0.0,
                    similarity,
                ),
            ),
            is_new_speaker=True,
            is_speaker_change=False,
            previous_speaker_id=None,
            timestamp=timestamp,
            metadata=dict(
                metadata or {}
            ),
        )

    # =================================================================
    # SPEAKER SIMILARITY
    # =================================================================

    def _speaker_similarity(
        self,
        embedding: np.ndarray,
        speaker: Speaker,
    ) -> float:
        """
        Calcule la similarité d'un embedding avec un speaker.

        Plusieurs embeddings historiques peuvent être utilisés.

        CORRECTIF : cette méthode retournait max(similarities), la
        MEILLEURE similarité sur le prototype ET tout l'historique.

        Conséquence démontrée par simulation : un seul segment
        ambigu — parole superposée, début ou fin de tour, bruit —
        était accepté pour le locuteur A, puis mémorisé tel quel dans
        son historique. Il devenait alors une référence, et TOUS les
        segments du locuteur B passaient ensuite le seuil face à
        cette seule référence. Les deux locuteurs fusionnaient
        définitivement.

        Le max est d'autant plus fragile que l'historique grandit :
        avec 11 vecteurs mémorisés, la probabilité qu'au moins un
        dépasse le seuil par hasard devient dix fois plus grande
        qu'avec un seul.

        On retourne désormais la MOYENNE DES MEILLEURES similarités.
        Un vecteur isolé ne peut plus décider seul, mais une vraie
        variation de voix reste tolérée par les suivants.
        """

        similarities: List[float] = []

        prototype_similarity: float = -2.0

        # -------------------------------------------------------------
        # Embedding principal  (le prototype)
        # -------------------------------------------------------------

        if speaker.embedding is not None:

            try:

                prototype_similarity = (
                    self._cosine_similarity(
                        embedding,
                        speaker.embedding,
                    )
                )

            except Exception:

                logger.warning(
                    "Invalid stored embedding for "
                    "speaker=%s",
                    speaker.speaker_id,
                )

        # -------------------------------------------------------------
        # Historique
        # -------------------------------------------------------------

        for historical_embedding in (
            speaker.embedding_history
        ):

            try:

                similarity = (
                    self._cosine_similarity(
                        embedding,
                        historical_embedding,
                    )
                )

                similarities.append(
                    similarity
                )

            except Exception:

                continue

        # -------------------------------------------------------------
        # Score
        # -------------------------------------------------------------
        #
        # Le prototype est une moyenne glissante : il represente
        # l'identite stable du locuteur, et aucun segment isole ne peut
        # le detourner. On lui fait donc confiance directement.
        #
        # L'historique, lui, contient des vecteurs bruts. C'est de la
        # que venait la fusion irreversible : un seul segment ambigu
        # memorise servait ensuite de reference. On ne retient donc pas
        # son maximum mais la moyenne de ses meilleures valeurs, ce qui
        # exige plusieurs concordances et non une seule.
        #
        # Le score final est le meilleur des deux. Prendre la moyenne
        # des deux penaliserait a tort un locuteur qui n'a encore que
        # peu d'historique.

        historique = -2.0

        if similarities:

            similarities.sort(
                reverse=True,
            )

            retenues = similarities[
                :_SIMILARITES_RETENUES
            ]

            historique = sum(retenues) / len(retenues)

        meilleur = max(
            prototype_similarity,
            historique,
        )

        if meilleur <= -2.0:

            return -1.0

        return meilleur

    # =================================================================
    # UPDATE SPEAKER
    # =================================================================

    def _update_speaker(
        self,
        speaker: Speaker,
        embedding: np.ndarray,
        confidence: float,
        timestamp: Optional[float],
        apprendre: bool = True,
    ) -> None:
        """
        Met à jour le profil du speaker.

        CORRECTIF : le prototype et l'historique étaient mis à jour
        pour TOUT segment accepté, y compris un segment tout juste
        au-dessus du seuil ou attribuable à deux locuteurs. Ce
        segment devenait ensuite une référence, et absorbait les
        segments de l'autre locuteur.

        Quand apprendre vaut False, le segment est bien attribué au
        locuteur — les compteurs et la date de dernière activité sont
        mis à jour — mais il n'entre ni dans le prototype ni dans
        l'historique.
        """

        if apprendre:

            # ---------------------------------------------------------
            # Mise à jour du prototype
            # ---------------------------------------------------------

            speaker.embedding = (
                self._update_embedding(
                    speaker.embedding,
                    embedding,
                )
            )

            # ---------------------------------------------------------
            # Historique
            # ---------------------------------------------------------

            speaker.embedding_history.append(
                embedding.copy()
            )

        if (
            len(
                speaker.embedding_history
            )
            > MAX_EMBEDDING_HISTORY
        ):

            speaker.embedding_history = (
                speaker.embedding_history[
                    -MAX_EMBEDDING_HISTORY:
                ]
            )

        # -------------------------------------------------------------
        # Statistiques
        # -------------------------------------------------------------

        speaker.confidence = (
            max(
                0.0,
                min(
                    1.0,
                    float(confidence),
                ),
            )
        )

        speaker.segment_count += 1

        speaker.last_seen = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        speaker.active = True

    # =================================================================
    # COSINE SIMILARITY
    # =================================================================

    @staticmethod
    def _cosine_similarity(
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        """
        Calcule la similarité cosinus.
        """

        first = np.asarray(
            first,
            dtype=np.float32,
        ).reshape(-1)

        second = np.asarray(
            second,
            dtype=np.float32,
        ).reshape(-1)

        # -------------------------------------------------------------
        # Dimensions incompatibles
        # -------------------------------------------------------------

        if first.shape != second.shape:

            return -1.0

        first_norm = np.linalg.norm(
            first
        )

        second_norm = np.linalg.norm(
            second
        )

        if (
            first_norm <= 0.0
            or second_norm <= 0.0
        ):

            return -1.0

        similarity = float(
            np.dot(
                first,
                second,
            )
            / (
                first_norm
                * second_norm
            )
        )

        if not np.isfinite(
            similarity
        ):

            return -1.0

        return max(
            -1.0,
            min(
                1.0,
                similarity,
            ),
        )

    # =================================================================
    # NORMALIZE EMBEDDING
    # =================================================================

    @staticmethod
    def _normalize_embedding(
        embedding: Any,
    ) -> np.ndarray:
        """
        Normalise un embedding vocal.
        """

        if embedding is None:

            raise DiarizationError(
                "Embedding cannot be None."
            )

        if isinstance(
            embedding,
            (
                str,
                bytes,
            ),
        ):

            raise DiarizationError(
                "Embedding must be numeric."
            )

        try:

            array = np.asarray(
                embedding,
                dtype=np.float32,
            )

        except Exception as exc:

            raise DiarizationError(
                "Invalid speaker embedding."
            ) from exc

        if array.size == 0:

            raise DiarizationError(
                "Speaker embedding cannot be empty."
            )

        array = array.reshape(-1)

        if not np.all(
            np.isfinite(array)
        ):

            raise DiarizationError(
                "Speaker embedding contains "
                "invalid numeric values."
            )

        norm = np.linalg.norm(
            array
        )

        if norm <= 0.0:

            raise DiarizationError(
                "Speaker embedding has zero norm."
            )

        return (
            array / norm
        ).astype(
            np.float32
        )

    # =================================================================
    # UPDATE EMBEDDING
    # =================================================================

    @staticmethod
    def _update_embedding(
        current: np.ndarray,
        new: np.ndarray,
    ) -> np.ndarray:
        """
        Mise à jour progressive du prototype vocal.
        """

        current = np.asarray(
            current,
            dtype=np.float32,
        ).reshape(-1)

        new = np.asarray(
            new,
            dtype=np.float32,
        ).reshape(-1)

        if current.shape != new.shape:

            return new.copy()

        updated = (
            0.70 * current
            + 0.30 * new
        )

        norm = np.linalg.norm(
            updated
        )

        if norm <= 0.0:

            return new.copy()

        return (
            updated / norm
        ).astype(
            np.float32
        )

    # =================================================================
    # THRESHOLD
    # =================================================================

    def _get_similarity_threshold(
        self,
    ) -> float:
        """
        Récupère le seuil d'identification.

        La configuration reste prioritaire.

        Si aucune configuration exploitable n'est disponible,
        le seuil de compatibilité est 0.55.
        """

        possible_names = (
            "speaker_similarity_threshold",
            "speaker_threshold",
            "embedding_similarity_threshold",
            "similarity_threshold",
        )

        for name in possible_names:

            value = getattr(
                self.config,
                name,
                None,
            )

            if value is None:
                continue

            try:

                value = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            if not np.isfinite(
                value
            ):

                continue

            if 0.0 <= value <= 1.0:

                return value

        return DEFAULT_SIMILARITY_THRESHOLD

    # =================================================================
    # CONFIDENCE
    # =================================================================

    @staticmethod
    def _similarity_to_confidence(
        similarity: float,
        threshold: float,
    ) -> float:
        """
        Convertit la similarité en confiance.

        CORRECTIF : la formule était discontinue au seuil. Sous le
        seuil elle renvoyait la similarité brute, au-dessus elle
        renvoyait (similarité - seuil) / (1 - seuil). Avec un seuil
        de 0,65 :

            similarité 0,6500  ->  confiance 0,6500
            similarité 0,6501  ->  confiance 0,0003

        Un match légèrement meilleur recevait donc une confiance
        deux mille fois plus faible. La branche du bas n'était de
        toute façon jamais atteinte par un vrai match, puisqu'un
        match exige similarité >= seuil.

        La formule est maintenant unique et monotone : 0 au seuil,
        1 à la similarité maximale, et 0 en dessous du seuil.
        """

        similarity = max(
            -1.0,
            min(
                1.0,
                float(similarity),
            ),
        )

        threshold = max(
            0.0,
            min(
                1.0,
                float(threshold),
            ),
        )

        if similarity <= threshold:

            return 0.0

        if threshold >= 1.0:

            return 1.0

        confidence = (
            similarity - threshold
        ) / (
            1.0 - threshold
        )

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

    # =================================================================
    # GENERATE SPEAKER ID
    # =================================================================

    @staticmethod
    def _generate_speaker_id() -> str:
        """
        Génère un identifiant speaker local à la session.
        """

        return (
            "speaker_"
            + uuid.uuid4().hex[:8]
        )

    # =================================================================
    # CURRENT SPEAKER
    # =================================================================

    def get_current_speaker(
        self,
        session_id: str,
    ) -> Optional[str]:
        """
        Retourne le speaker courant.
        """

        session_id = self._validate_session_id(
            session_id
        )

        with self._lock:

            return self._current_speaker.get(
                session_id
            )

    # =================================================================
    # SET CURRENT SPEAKER
    # =================================================================

    def set_current_speaker(
        self,
        session_id: str,
        speaker_id: Optional[str],
    ) -> bool:
        """
        Définit manuellement le speaker courant.
        """

        session_id = self._validate_session_id(
            session_id
        )

        with self._lock:

            self.create_session(
                session_id
            )

            if speaker_id is None:

                self._current_speaker[
                    session_id
                ] = None

                return True

            if not isinstance(
                speaker_id,
                str,
            ):

                raise SessionError(
                    "speaker_id must be a string."
                )

            speaker_id = speaker_id.strip()

            if not speaker_id:

                raise SessionError(
                    "speaker_id cannot be empty."
                )

            if speaker_id not in self._speakers[
                session_id
            ]:

                return False

            self._current_speaker[
                session_id
            ] = speaker_id

            return True

    # =================================================================
    # GET SPEAKER
    # =================================================================

    def get_speaker(
        self,
        session_id: str,
        speaker_id: str,
    ) -> Optional[Speaker]:
        """
        Retourne un speaker.
        """

        session_id = self._validate_session_id(
            session_id
        )

        speaker_id = (
            self._validate_speaker_id(
                speaker_id
            )
        )

        with self._lock:

            return self._speakers.get(
                session_id,
                {},
            ).get(
                speaker_id
            )

    # =================================================================
    # LIST SPEAKERS
    # =================================================================

    def list_speakers(
        self,
        session_id: str,
    ) -> List[Speaker]:
        """
        Retourne tous les speakers d'une session.
        """

        session_id = self._validate_session_id(
            session_id
        )

        with self._lock:

            return list(
                self._speakers.get(
                    session_id,
                    {},
                ).values()
            )

    # =================================================================
    # GET SPEAKERS
    # =================================================================

    def get_speakers(
        self,
        session_id: str,
    ) -> List[Speaker]:
        """
        Alias de list_speakers().
        """

        return self.list_speakers(
            session_id
        )

    # =================================================================
    # SPEAKER COUNT
    # =================================================================

    def speaker_count(
        self,
        session_id: str,
    ) -> int:
        """
        Nombre de speakers d'une session.
        """

        return len(
            self.list_speakers(
                session_id
            )
        )

    # =================================================================
    # REMOVE SPEAKER
    # =================================================================

    def remove_speaker(
        self,
        session_id: str,
        speaker_id: str,
    ) -> bool:
        """
        Supprime un speaker.
        """

        session_id = self._validate_session_id(
            session_id
        )

        speaker_id = (
            self._validate_speaker_id(
                speaker_id
            )
        )

        with self._lock:

            speakers = self._speakers.get(
                session_id
            )

            if not speakers:

                return False

            if speaker_id not in speakers:

                return False

            del speakers[
                speaker_id
            ]

            if (
                self._current_speaker.get(
                    session_id
                )
                == speaker_id
            ):

                self._current_speaker[
                    session_id
                ] = None

            return True

    # =================================================================
    # DELETE SPEAKER
    # =================================================================

    def delete_speaker(
        self,
        session_id: str,
        speaker_id: str,
    ) -> bool:
        """
        Alias historique.
        """

        return self.remove_speaker(
            session_id=session_id,
            speaker_id=speaker_id,
        )

    # =================================================================
    # UPDATE SPEECH
    # =================================================================

    def update_speech(
        self,
        session_id: str,
        speaker_id: str,
        duration: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Met à jour les statistiques vocales.
        """

        session_id = self._validate_session_id(
            session_id
        )

        speaker_id = (
            self._validate_speaker_id(
                speaker_id
            )
        )

        try:

            duration = float(
                duration
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise DiarizationError(
                "duration must be numeric."
            ) from exc

        if not np.isfinite(
            duration
        ) or duration < 0.0:

            raise DiarizationError(
                "duration must be >= 0."
            )

        with self._lock:

            speaker = self._speakers.get(
                session_id,
                {},
            ).get(
                speaker_id
            )

            if speaker is None:

                raise SessionError(
                    "Speaker not found."
                )

            if timestamp is not None:

                try:

                    timestamp = float(
                        timestamp
                    )

                except (
                    TypeError,
                    ValueError,
                ) as exc:

                    raise DiarizationError(
                        "timestamp must be numeric."
                    ) from exc

                if timestamp < 0.0:

                    raise DiarizationError(
                        "timestamp must be >= 0."
                    )

                speaker.last_seen = timestamp

            else:

                speaker.last_seen = time.time()

            speaker.active = True

    # =================================================================
    # MARK ALL INACTIVE
    # =================================================================

    def mark_all_inactive(
        self,
        session_id: str,
    ) -> None:
        """
        Marque tous les speakers comme inactifs.
        """

        session_id = self._validate_session_id(
            session_id
        )

        with self._lock:

            speakers = self._speakers.get(
                session_id,
                {},
            )

            for speaker in speakers.values():

                speaker.active = False

    # =================================================================
    # CLEAR SESSION
    # =================================================================

    def clear_session(
        self,
        session_id: str,
    ) -> None:
        """
        Supprime tous les speakers d'une session.
        """

        self.reset_session(
            session_id
        )

    # =================================================================
    # DELETE SESSION
    # =================================================================

    def delete_session(
        self,
        session_id: str,
    ) -> None:
        """
        Alias de remove_session.
        """

        self.remove_session(
            session_id
        )

    # =================================================================
    # RESET
    # =================================================================

    def reset(self) -> None:
        """
        Réinitialise toutes les sessions.
        """

        with self._lock:

            self._speakers.clear()

            self._current_speaker.clear()

            self._identified_segments = 0

            self._created_speakers = 0

    # =================================================================
    # HEALTH
    # =================================================================

    def health(
        self,
    ) -> Dict[str, Any]:
        """
        Health check.
        """

        try:

            with self._lock:

                session_count = len(
                    self._speakers
                )

                speaker_count = sum(
                    len(
                        speakers
                    )
                    for speakers
                    in self._speakers.values()
                )

            return {
                "service":
                    "speakeridentity_speaker_manager",
                "status":
                    "healthy",
                "sessions":
                    session_count,
                "speakers":
                    speaker_count,
            }

        except Exception as exc:

            logger.exception(
                "SpeakerManager health check failed."
            )

            return {
                "service":
                    "speakeridentity_speaker_manager",
                "status":
                    "unhealthy",
                "error":
                    str(exc),
            }

    # =================================================================
    # STATISTICS
    # =================================================================

    def statistics(
        self,
    ) -> Dict[str, Any]:
        """
        Retourne les statistiques.
        """

        with self._lock:

            sessions = len(
                self._speakers
            )

            speakers = sum(
                len(items)
                for items
                in self._speakers.values()
            )

            segments = sum(
                speaker.segment_count
                for items
                in self._speakers.values()
                for speaker
                in items.values()
            )

        return {
            "component":
                "speaker_manager",
            "sessions":
                sessions,
            "speakers":
                speakers,
            "identified_segments":
                segments,
            "created_speakers":
                self._created_speakers,
        }

    # =================================================================
    # LENGTH
    # =================================================================

    def __len__(self) -> int:
        """
        Nombre total de speakers.
        """

        with self._lock:

            return sum(
                len(items)
                for items
                in self._speakers.values()
            )

    # =================================================================
    # VALIDATE SESSION
    # =================================================================

    @staticmethod
    def _validate_session_id(
        session_id: Optional[str],
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

    # =================================================================
    # VALIDATE SPEAKER
    # =================================================================

    @staticmethod
    def _validate_speaker_id(
        speaker_id: Optional[str],
    ) -> str:
        """
        Valide un identifiant speaker.
        """

        if not isinstance(
            speaker_id,
            str,
        ):

            raise SessionError(
                "speaker_id must be a string."
            )

        speaker_id = speaker_id.strip()

        if not speaker_id:

            raise SessionError(
                "speaker_id cannot be empty."
            )

        return speaker_id


# ====================================================================
# FACTORY
# ====================================================================

def create_speaker_manager(
    config: Optional[
        DiarizationConfig
    ] = None,
) -> SpeakerManager:
    """
    Factory du SpeakerManager.
    """

    return SpeakerManager(
        config=(
            config
            if config is not None
            else default_config
        )
    )


# ====================================================================
# GLOBAL INSTANCE
# ====================================================================

speaker_manager = create_speaker_manager()


# ====================================================================
# PUBLIC API
# ====================================================================

__all__ = [
    "UNKNOWN_SPEAKER_ID",
    "Speaker",
    "SpeakerIdentificationResult",
    "SpeakerManager",
    "create_speaker_manager",
    "speaker_manager",
]