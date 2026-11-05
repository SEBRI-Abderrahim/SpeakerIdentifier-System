"""
====================================================================
SpeakerIdentity IA - Diarization Segment Manager
====================================================================

Projet       : SpeakerIdentity IA
Module       : Diarization
Fichier      : diarization/segment_manager.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 2.0.6
Date         : 14/08/2026

Description
------------

Gestion des segments audio issus du VAD.

Responsabilités :

    - création des segments ;
    - accumulation des frames audio ;
    - gestion du temps audio par session ;
    - fermeture des segments ;
    - durée minimale / maximale ;
    - padding temporel ;
    - stockage des segments terminés ;
    - flush des segments actifs ;
    - statistiques ;
    - health check ;
    - propagation de la confiance fournie par le VAD.

Format audio interne SpeakerIdentity :

    PCM16
    16000 Hz
    Mono

====================================================================
"""

from __future__ import annotations

import threading
import time
import uuid

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import (
    SAMPLE_RATE,
    CHANNELS,
    SAMPLE_WIDTH,
    MIN_SEGMENT_DURATION_MS,
    MAX_SEGMENT_DURATION_MS,
    SEGMENT_PADDING_MS,
    MAX_SEGMENTS_PER_SESSION,
)

from .exceptions import (
    AudioFormatError,
    SessionError,
)

from .metrics import metrics


# ====================================================================
# CONSTANTS
# ====================================================================

_PCM16_BYTES_PER_SAMPLE = 2


# CORRECTIF : la duree d'un segment est une somme de flottants de 0,01 s.
# Apres 1000 frames elle vaut 9.999999999999831, jamais exactement 10.0.
# Les comparaisons a max_segment_duration n'etaient donc JAMAIS vraies :
# sur de la parole continue, le segment actif restait bloque a 10 s, plus
# aucun segment n'etait emis, et tout l'audio suivant etait detruit en
# silence. Sans segment, pas d'embedding, donc pas d'identification :
# c'est la cause principale du "il ne detecte qu'un seul interlocuteur".
#
# Verifie par reproduction : 14 s de parole continue produisaient
# 0 segment. Avec cette tolerance, elles en produisent bien 1 par
# tranche de 10 s.
#
# 1 microseconde : tres au-dessus du bruit d'arrondi (~1e-13 s) et tres
# en dessous d'un echantillon a 16 kHz (62,5 microsecondes).

_DUREE_EPSILON = 1e-6


# ====================================================================
# ACTIVE SEGMENT
# ====================================================================

@dataclass
class ActiveSegment:
    """
    Segment audio actuellement en construction.
    """

    session_id: str

    start: float

    last_activity: float

    audio: bytearray

    speech_start: float

    speech_end: Optional[float] = None

    sequence: int = 0

    created_at: float = 0.0

    last_was_speech: bool = True

    speech_frames: int = 0

    silence_frames: int = 0

    audio_duration: float = 0.0

    confidence_sum: float = 0.0

    confidence_count: int = 0

    def __post_init__(self) -> None:
        """
        Initialise les valeurs dérivées du segment.
        """

        if self.created_at <= 0.0:
            self.created_at = time.time()

        if self.audio_duration <= 0.0:
            bytes_per_second = (
                SAMPLE_RATE
                * CHANNELS
                * SAMPLE_WIDTH
            )

            if bytes_per_second > 0:
                self.audio_duration = (
                    len(self.audio)
                    / bytes_per_second
                )

    @property
    def duration(self) -> float:
        """
        Durée temporelle du segment actif.
        """

        end = self.last_activity

        if self.speech_end is not None:
            end = max(
                end,
                self.speech_end,
            )

        return max(
            0.0,
            end - self.start,
        )

    @property
    def audio_size(self) -> int:
        """
        Taille audio PCM en octets.
        """

        return len(self.audio)

    @property
    def confidence(self) -> float:
        """
        Confiance moyenne réellement fournie par le VAD.

        Si aucune confiance n'a été fournie par le VAD,
        retourne 0.0 et non une valeur artificielle.
        """

        if self.confidence_count <= 0:
            return 0.0

        confidence = (
            self.confidence_sum
            / self.confidence_count
        )

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )


# ====================================================================
# SEGMENT RESULT
# ====================================================================

@dataclass
class SegmentResult:
    """
    Segment finalisé.
    """

    segment_id: str

    session_id: str

    start: float

    end: float

    audio: bytes

    confidence: float = 0.0

    sequence: int = 0

    metadata: Optional[Dict[str, Any]] = None

    # -----------------------------------------------------------------
    # CORRECTIF : le segment ne portait PAS l'identifiant du locuteur.
    #
    # Le pipeline essayait bien de le poser, mais derriere un garde
    #
    #     if hasattr(segment, "speaker_id"): segment.speaker_id = ...
    #
    # qui n'etait jamais vrai, puisque le champ n'existait pas. L'appel
    # etait donc ignore en silence, et l'identifiant ne vivait que dans
    # les evenements. Le navigateur, lui, lit segment.speaker_id : il
    # affichait donc "speaker_unknown" pour absolument tous les
    # segments, y compris quand l'identification avait parfaitement
    # fonctionne.
    #
    # confidence reste la confiance du VAD. La confiance de
    # l'identification est un tout autre indicateur : elle a son propre
    # champ, au lieu d'ecraser le premier.
    # -----------------------------------------------------------------

    speaker_id: Optional[str] = None

    speaker_confidence: float = 0.0

    # Similarite cosinus avec le locuteur retenu. Publiee pour pouvoir
    # regler le seuil sur des mesures et non a l'aveugle.
    speaker_similarity: float = 0.0

    # Seuil applique au moment de la decision. Publie pour qu'on puisse
    # lire, a l'ecran, POURQUOI un segment a matche ou non.
    speaker_threshold: float = 0.0

    is_new_speaker: bool = False

    @property
    def duration(self) -> float:
        """
        Durée temporelle du segment.
        """

        return max(
            0.0,
            self.end - self.start,
        )

    @property
    def audio_size(self) -> int:
        """
        Taille du PCM associé au segment.
        """

        return len(self.audio)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit le résultat en dictionnaire sérialisable.
        """

        try:
            confidence = float(
                self.confidence
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        try:
            start = float(self.start)
        except (
            TypeError,
            ValueError,
        ):
            start = 0.0

        try:
            end = float(self.end)
        except (
            TypeError,
            ValueError,
        ):
            end = start

        start = max(
            0.0,
            start,
        )

        end = max(
            start,
            end,
        )

        return {
            "segment_id": self.segment_id,
            "session_id": self.session_id,
            "start": round(
                start,
                3,
            ),
            "end": round(
                end,
                3,
            ),
            "duration": round(
                max(
                    0.0,
                    end - start,
                ),
                3,
            ),
            "audio_size": self.audio_size,
            "confidence": round(
                confidence,
                4,
            ),
            "speaker_id": self.speaker_id,
            "speaker_confidence": round(
                max(
                    0.0,
                    min(
                        1.0,
                        float(
                            self.speaker_confidence or 0.0
                        ),
                    ),
                ),
                4,
            ),
            "speaker_similarity": round(
                float(
                    self.speaker_similarity or 0.0
                ),
                4,
            ),
            "speaker_threshold": round(
                float(
                    self.speaker_threshold or 0.0
                ),
                4,
            ),
            "is_new_speaker": bool(
                self.is_new_speaker
            ),
            "sequence": self.sequence,
            "metadata": dict(
                self.metadata or {}
            ),
        }


# ====================================================================
# SEGMENT MANAGER
# ====================================================================

class SegmentManager:
    """
    Gestionnaire des segments de diarisation.

    Une session possède :

        - un segment actif maximum ;
        - une chronologie audio ;
        - une liste de segments terminés ;
        - un compteur de séquence.

    Le verrou interne protège toutes les structures
    partagées entre sessions.
    """

    def __init__(
        self,
        *,
        config: Optional[Any] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        sample_width: Optional[int] = None,
        min_segment_duration_ms: Optional[int] = None,
        max_segment_duration_ms: Optional[int] = None,
        segment_padding_ms: Optional[int] = None,
        max_segments_per_session: Optional[int] = None,
    ) -> None:

        self.config = config

        self.sample_rate = int(
            self._config_value(
                config,
                "sample_rate",
                "SAMPLE_RATE",
                SAMPLE_RATE,
                explicit=sample_rate,
            )
        )

        self.channels = int(
            self._config_value(
                config,
                "channels",
                "CHANNELS",
                CHANNELS,
                explicit=channels,
            )
        )

        self.sample_width = int(
            self._config_value(
                config,
                "sample_width",
                "SAMPLE_WIDTH",
                SAMPLE_WIDTH,
                explicit=sample_width,
            )
        )

        self.min_segment_duration_ms = int(
            self._config_value(
                config,
                "min_segment_duration_ms",
                "MIN_SEGMENT_DURATION_MS",
                MIN_SEGMENT_DURATION_MS,
                explicit=min_segment_duration_ms,
            )
        )

        self.max_segment_duration_ms = int(
            self._config_value(
                config,
                "max_segment_duration_ms",
                "MAX_SEGMENT_DURATION_MS",
                MAX_SEGMENT_DURATION_MS,
                explicit=max_segment_duration_ms,
            )
        )

        self.segment_padding_ms = int(
            self._config_value(
                config,
                "segment_padding_ms",
                "SEGMENT_PADDING_MS",
                SEGMENT_PADDING_MS,
                explicit=segment_padding_ms,
            )
        )

        self.max_segments_per_session = int(
            self._config_value(
                config,
                "max_segments_per_session",
                "MAX_SEGMENTS_PER_SESSION",
                MAX_SEGMENTS_PER_SESSION,
                explicit=max_segments_per_session,
            )
        )

        self._validate_configuration()

        self._active_segments: Dict[
            str,
            ActiveSegment,
        ] = {}

        self._completed_segments: Dict[
            str,
            List[SegmentResult],
        ] = {}

        self._sequence: Dict[
            str,
            int,
        ] = {}

        self._session_time: Dict[
            str,
            float,
        ] = {}

        self._lock = threading.RLock()

    # =================================================================
    # CONFIGURATION
    # =================================================================

    @staticmethod
    def _config_value(
        config: Optional[Any],
        lowercase_name: str,
        uppercase_name: str,
        default: Any,
        *,
        explicit: Any = None,
    ) -> Any:
        """
        Récupère une valeur depuis :

            1. override explicite ;
            2. dictionnaire ;
            3. attribut lowercase ;
            4. attribut uppercase ;
            5. valeur par défaut.
        """

        if explicit is not None:
            return explicit

        if config is None:
            return default

        if isinstance(config, dict):

            value = config.get(
                lowercase_name
            )

            if value is not None:
                return value

            value = config.get(
                uppercase_name
            )

            if value is not None:
                return value

        try:
            value = getattr(
                config,
                lowercase_name,
                None,
            )

            if value is not None:
                return value

        except Exception:
            pass

        try:
            value = getattr(
                config,
                uppercase_name,
                None,
            )

            if value is not None:
                return value

        except Exception:
            pass

        return default

    # =================================================================
    # CONFIDENCE
    # =================================================================

    @staticmethod
    def _extract_confidence(
        vad_result: Any,
    ) -> float:
        """
        Extrait la confiance réellement fournie par le VAD.

        Formats supportés :

            - float / int ;
            - {"confidence": value} ;
            - {"score": value} ;
            - {"probability": value} ;
            - objet .confidence ;
            - objet .score ;
            - objet .probability.

        Aucune confiance n'est inventée.

        Si aucune valeur exploitable n'est disponible,
        retourne 0.0.
        """

        if vad_result is None:
            return 0.0

        value: Any = None

        if isinstance(
            vad_result,
            (
                int,
                float,
            ),
        ):
            value = vad_result

        elif isinstance(
            vad_result,
            dict,
        ):

            for key in (
                "confidence",
                "score",
                "probability",
            ):

                if key in vad_result:
                    value = vad_result.get(
                        key
                    )

                    if value is not None:
                        break

        else:

            for attribute in (
                "confidence",
                "score",
                "probability",
            ):

                try:
                    value = getattr(
                        vad_result,
                        attribute,
                        None,
                    )
                except Exception:
                    value = None

                if value is not None:
                    break

        try:
            confidence = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        if confidence != confidence:
            return 0.0

        if confidence < 0.0:
            return 0.0

        if confidence > 1.0:
            return 1.0

        return confidence

    # =================================================================
    # VALIDATION
    # =================================================================

    def _validate_configuration(self) -> None:
        """
        Vérifie que la configuration est compatible
        avec le format audio SpeakerIdentity.
        """

        if self.sample_rate <= 0:
            raise AudioFormatError(
                "sample_rate must be greater than zero."
            )

        if self.sample_rate != SAMPLE_RATE:
            raise AudioFormatError(
                f"SpeakerIdentity diarization requires "
                f"{SAMPLE_RATE} Hz audio."
            )

        if self.channels != CHANNELS:
            raise AudioFormatError(
                "SpeakerIdentity diarization requires mono audio."
            )

        if self.sample_width != _PCM16_BYTES_PER_SAMPLE:
            raise AudioFormatError(
                "SpeakerIdentity diarization requires PCM16."
            )

        if self.min_segment_duration_ms < 0:
            raise ValueError(
                "min_segment_duration_ms must be >= 0."
            )

        if self.max_segment_duration_ms <= 0:
            raise ValueError(
                "max_segment_duration_ms must be > 0."
            )

        if (
            self.max_segment_duration_ms
            < self.min_segment_duration_ms
        ):
            raise ValueError(
                "max_segment_duration_ms must be "
                "greater than or equal to "
                "min_segment_duration_ms."
            )

        if self.segment_padding_ms < 0:
            raise ValueError(
                "segment_padding_ms must be >= 0."
            )

        if self.max_segments_per_session < 1:
            raise ValueError(
                "max_segments_per_session must be >= 1."
            )

    # =================================================================
    # DURATIONS
    # =================================================================

    @property
    def min_segment_duration(self) -> float:
        """
        Durée minimale en secondes.
        """

        return (
            self.min_segment_duration_ms
            / 1000.0
        )

    @property
    def max_segment_duration(self) -> float:
        """
        Durée maximale en secondes.
        """

        return (
            self.max_segment_duration_ms
            / 1000.0
        )

    @property
    def segment_padding(self) -> float:
        """
        Padding temporel en secondes.
        """

        return (
            self.segment_padding_ms
            / 1000.0
        )

    # =================================================================
    # SESSION
    # =================================================================

    def create_session(
        self,
        session_id: str,
    ) -> None:
        """
        Initialise une session si nécessaire.
        """

        session_id = self._normalize_session_id(
            session_id
        )

        with self._lock:

            self._completed_segments.setdefault(
                session_id,
                [],
            )

            self._sequence.setdefault(
                session_id,
                0,
            )

            self._session_time.setdefault(
                session_id,
                0.0,
            )

    def remove_session(
        self,
        session_id: str,
    ) -> None:
        """
        Supprime toutes les données internes
        d'une session.
        """

        session_id = self._normalize_session_id(
            session_id
        )

        with self._lock:

            self._active_segments.pop(
                session_id,
                None,
            )

            self._completed_segments.pop(
                session_id,
                None,
            )

            self._sequence.pop(
                session_id,
                None,
            )

            self._session_time.pop(
                session_id,
                None,
            )

    def _normalize_session_id(
        self,
        session_id: str,
    ) -> str:
        """
        Normalise et valide l'identifiant de session.
        """

        self._validate_session_id(
            session_id
        )

        return session_id.strip()

    @staticmethod
    def _validate_session_id(
        session_id: str,
    ) -> None:
        """
        Vérifie l'identifiant de session.
        """

        if not isinstance(
            session_id,
            str,
        ):
            raise SessionError(
                "session_id must be a string."
            )

        if not session_id.strip():
            raise SessionError(
                "session_id must be a non-empty string."
            )

    # =================================================================
    # AUDIO VALIDATION
    # =================================================================

    def _validate_audio(
        self,
        audio: bytes,
    ) -> bytes:
        """
        Valide une donnée audio PCM16 mono.
        """

        if not isinstance(
            audio,
            (
                bytes,
                bytearray,
                memoryview,
            ),
        ):
            raise AudioFormatError(
                "Audio must be bytes-like."
            )

        audio_bytes = bytes(
            audio
        )

        if not audio_bytes:
            return b""

        bytes_per_sample = (
            self.sample_width
            * self.channels
        )

        if bytes_per_sample <= 0:
            raise AudioFormatError(
                "Invalid audio format configuration."
            )

        if (
            len(audio_bytes)
            % bytes_per_sample
            != 0
        ):
            raise AudioFormatError(
                "Audio byte length is not aligned "
                "with PCM16 mono."
            )

        return audio_bytes

    def _audio_duration(
        self,
        audio: bytes,
    ) -> float:
        """
        Retourne la durée d'un buffer audio en secondes.
        """

        bytes_per_second = (
            self.sample_rate
            * self.channels
            * self.sample_width
        )

        if bytes_per_second <= 0:
            return 0.0

        return (
            len(audio)
            / bytes_per_second
        )

    def _duration_to_bytes(
        self,
        duration: float,
    ) -> int:
        """
        Convertit une durée en octets PCM alignés.
        """

        if duration <= 0.0:
            return 0

        bytes_per_second = (
            self.sample_rate
            * self.channels
            * self.sample_width
        )

        alignment = (
            self.sample_width
            * self.channels
        )

        if (
            bytes_per_second <= 0
            or alignment <= 0
        ):
            return 0

        raw_size = int(
            duration
            * bytes_per_second
        )

        return (
            raw_size
            // alignment
        ) * alignment

    # =================================================================
    # SEQUENCE
    # =================================================================

    def _next_sequence(
        self,
        session_id: str,
    ) -> int:
        """
        Retourne le prochain numéro de séquence.
        """

        current = self._sequence.get(
            session_id,
            0,
        )

        current += 1

        self._sequence[
            session_id
        ] = current

        return current

    # =================================================================
    # PROCESS FRAME
    # =================================================================

    def process_frame(
        self,
        *,
        session_id: str,
        frame: bytes,
        is_speech: bool,
        vad_result: Any = None,
    ) -> List[SegmentResult]:
        """
        Traite une frame audio issue du VAD.

        SPEECH :
            crée ou poursuit un segment.

        SILENCE :
            conserve la chronologie et termine le segment
            lors de la transition SPEECH -> SILENCE.

        La confiance est extraite du résultat VAD et
        accumulée pendant le segment.

        Une frame dépassant la durée maximale est découpée
        afin d'éviter toute perte audio.
        """

        session_id = self._normalize_session_id(
            session_id
        )

        frame = self._validate_audio(
            frame
        )

        if not isinstance(
            is_speech,
            bool,
        ):
            raise TypeError(
                "is_speech must be a boolean."
            )

        confidence = (
            self._extract_confidence(
                vad_result
            )
        )

        if not frame:
            return []

        with self._lock:

            self.create_session(
                session_id
            )

            timestamp = max(
                0.0,
                self._session_time.get(
                    session_id,
                    0.0,
                ),
            )

            duration = self._audio_duration(
                frame
            )

            results: List[
                SegmentResult
            ] = []

            # ========================================================
            # SPEECH
            # ========================================================

            if is_speech:

                offset = 0

                while offset < len(frame):

                    current_timestamp = (
                        timestamp
                        + self._audio_duration(
                            frame[:offset]
                        )
                    )

                    active = (
                        self._active_segments.get(
                            session_id
                        )
                    )

                    if active is None:

                        self.start_segment(
                            session_id=session_id,
                            timestamp=current_timestamp,
                            confidence=confidence,
                        )

                        active = (
                            self._active_segments.get(
                                session_id
                            )
                        )

                        if active is None:
                            break

                    remaining_duration = max(
                        0.0,
                        self.max_segment_duration
                        - active.duration,
                    )

                    if remaining_duration <= _DUREE_EPSILON:

                        finalized = (
                            self._finalize_active_segment(
                                session_id
                            )
                        )

                        results.extend(
                            finalized
                        )

                        continue

                    remaining_audio = (
                        frame[offset:]
                    )

                    remaining_audio_duration = (
                        self._audio_duration(
                            remaining_audio
                        )
                    )

                    append_duration = min(
                        remaining_duration,
                        remaining_audio_duration,
                    )

                    append_bytes = (
                        self._duration_to_bytes(
                            append_duration
                        )
                    )

                    if append_bytes <= 0:
                        break

                    chunk = remaining_audio[
                        :append_bytes
                    ]

                    appended_results = (
                        self.append_audio(
                            session_id=session_id,
                            audio=chunk,
                            timestamp=current_timestamp,
                            is_speech=True,
                            confidence=confidence,
                        )
                    )

                    results.extend(
                        appended_results
                    )

                    offset += len(chunk)

                    current_end = (
                        current_timestamp
                        + self._audio_duration(
                            chunk
                        )
                    )

                    self._session_time[
                        session_id
                    ] = max(
                        self._session_time.get(
                            session_id,
                            0.0,
                        ),
                        current_end,
                    )

                    if (
                        len(chunk)
                        == 0
                    ):
                        break

                self._session_time[
                    session_id
                ] = max(
                    self._session_time.get(
                        session_id,
                        0.0,
                    ),
                    timestamp + duration,
                )

                return results

            # ========================================================
            # SILENCE
            # ========================================================

            active = (
                self._active_segments.get(
                    session_id
                )
            )

            silence_end = (
                timestamp + duration
            )

            if active is None:

                self._session_time[
                    session_id
                ] = max(
                    self._session_time.get(
                        session_id,
                        0.0,
                    ),
                    silence_end,
                )

                return []

            active.last_activity = max(
                active.last_activity,
                silence_end,
            )

            active.silence_frames += 1

            # ========================================================
            # SPEECH -> SILENCE
            # ========================================================

            if active.last_was_speech:

                active.speech_end = timestamp

                active.last_was_speech = False

                results.extend(
                    self._finalize_active_segment(
                        session_id
                    )
                )

            else:

                active.last_was_speech = False

            self._session_time[
                session_id
            ] = max(
                self._session_time.get(
                    session_id,
                    0.0,
                ),
                silence_end,
            )

            return results

    # =================================================================
    # COMPATIBILITY ALIAS
    # =================================================================

    def process(
        self,
        *,
        session_id: str,
        frame: bytes,
        is_speech: bool,
        vad_result: Any = None,
    ) -> List[SegmentResult]:
        """
        Alias de compatibilité pour process_frame().
        """

        return self.process_frame(
            session_id=session_id,
            frame=frame,
            is_speech=is_speech,
            vad_result=vad_result,
        )

    # =================================================================
    # START SEGMENT
    # =================================================================

    def start_segment(
        self,
        session_id: str,
        timestamp: float,
        audio: bytes = b"",
        confidence: float = 0.0,
    ) -> None:
        """
        Démarre un nouveau segment.

        Le paramètre audio est optionnel afin de conserver
        la compatibilité avec le pipeline existant.

        confidence correspond uniquement à la confiance
        réellement fournie par le VAD.
        """

        session_id = self._normalize_session_id(
            session_id
        )

        audio = self._validate_audio(
            audio
        )

        confidence = self._extract_confidence(
            confidence
        )

        try:
            timestamp = float(
                timestamp
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "timestamp must be numeric."
            ) from exc

        timestamp = max(
            0.0,
            timestamp,
        )

        with self._lock:

            self.create_session(
                session_id
            )

            if session_id in self._active_segments:
                return

            if (
                len(
                    self._completed_segments.get(
                        session_id,
                        [],
                    )
                )
                >= self.max_segments_per_session
            ):
                metrics.increment(
                    "errors"
                )
                return

            sequence = self._next_sequence(
                session_id
            )

            audio_duration = (
                self._audio_duration(
                    audio
                )
            )

            end = (
                timestamp
                + audio_duration
            )

            self._active_segments[
                session_id
            ] = ActiveSegment(
                session_id=session_id,
                start=timestamp,
                last_activity=end,
                audio=bytearray(
                    audio
                ),
                speech_start=timestamp,
                speech_end=(
                    end
                    if audio
                    else None
                ),
                sequence=sequence,
                last_was_speech=True,
                speech_frames=(
                    1
                    if audio
                    else 0
                ),
                silence_frames=0,
                audio_duration=audio_duration,
                confidence_sum=(
                    confidence
                    if audio
                    else 0.0
                ),
                confidence_count=(
                    1
                    if audio
                    else 0
                ),
            )

    # =================================================================
    # APPEND AUDIO
    # =================================================================

    def append_audio(
        self,
        session_id: str,
        audio: bytes,
        timestamp: float,
        *,
        is_speech: bool = True,
        confidence: float = 0.0,
    ) -> List[SegmentResult]:
        """
        Ajoute une frame au segment actif.

        Cette méthode ne laisse pas volontairement le segment
        dépasser la durée maximale.

        confidence correspond à la confiance fournie par le VAD.
        """

        session_id = self._normalize_session_id(
            session_id
        )

        audio = self._validate_audio(
            audio
        )

        confidence = self._extract_confidence(
            confidence
        )

        try:
            timestamp = float(
                timestamp
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "timestamp must be numeric."
            ) from exc

        timestamp = max(
            0.0,
            timestamp,
        )

        if not isinstance(
            is_speech,
            bool,
        ):
            raise TypeError(
                "is_speech must be a boolean."
            )

        with self._lock:

            self.create_session(
                session_id
            )

            active = (
                self._active_segments.get(
                    session_id
                )
            )

            if active is None:

                if not is_speech:
                    return []

                self.start_segment(
                    session_id=session_id,
                    timestamp=timestamp,
                    audio=audio,
                    confidence=confidence,
                )

                return self._split_if_needed(
                    session_id
                )

            audio_duration = (
                self._audio_duration(
                    audio
                )
            )

            timestamp = max(
                timestamp,
                active.last_activity,
            )

            remaining_duration = max(
                0.0,
                self.max_segment_duration
                - active.duration,
            )

            if (
                audio_duration > 0.0
                and remaining_duration <= _DUREE_EPSILON
            ):
                return (
                    self._finalize_active_segment(
                        session_id
                    )
                )

            if (
                audio_duration > 0.0
                and audio_duration > remaining_duration
            ):

                allowed_bytes = (
                    self._duration_to_bytes(
                        remaining_duration
                    )
                )

                chunk = audio[
                    :allowed_bytes
                ]

                if chunk:

                    chunk_duration = (
                        self._audio_duration(
                            chunk
                        )
                    )

                    active.audio.extend(
                        chunk
                    )

                    active.audio_duration += (
                        chunk_duration
                    )

                    active.last_activity = max(
                        active.last_activity,
                        timestamp + chunk_duration,
                    )

                    if is_speech:

                        active.speech_end = (
                            timestamp
                            + chunk_duration
                        )

                        active.speech_frames += 1

                        active.last_was_speech = True

                        active.confidence_sum += (
                            confidence
                        )

                        active.confidence_count += 1

                    else:

                        active.silence_frames += 1

                        active.last_was_speech = False

                return (
                    self._finalize_active_segment(
                        session_id
                    )
                )

            if audio:

                active.audio.extend(
                    audio
                )

                active.audio_duration += (
                    audio_duration
                )

            audio_end = (
                timestamp
                + audio_duration
            )

            active.last_activity = max(
                active.last_activity,
                audio_end,
            )

            if is_speech:

                active.speech_end = (
                    audio_end
                )

                active.speech_frames += 1

                active.last_was_speech = True

                active.confidence_sum += (
                    confidence
                )

                active.confidence_count += 1

            else:

                active.silence_frames += 1

                active.last_was_speech = False

            return self._split_if_needed(
                session_id
            )

    # =================================================================
    # MAX SPLIT
    # =================================================================

    def _split_if_needed(
        self,
        session_id: str,
    ) -> List[SegmentResult]:
        """
        Finalise le segment lorsqu'il atteint
        sa durée maximale.
        """

        active = (
            self._active_segments.get(
                session_id
            )
        )

        if active is None:
            return []

        if (
            active.duration
            < self.max_segment_duration - _DUREE_EPSILON
        ):
            return []

        return self._finalize_active_segment(
            session_id
        )

    # =================================================================
    # END SEGMENT
    # =================================================================

    def end_segment(
        self,
        session_id: str,
        timestamp: float,
        *,
        force: bool = False,
    ) -> Optional[SegmentResult]:
        """
        Termine explicitement le segment actif.
        """

        session_id = self._normalize_session_id(
            session_id
        )

        try:
            timestamp = float(
                timestamp
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "timestamp must be numeric."
            ) from exc

        timestamp = max(
            0.0,
            timestamp,
        )

        with self._lock:

            active = (
                self._active_segments.get(
                    session_id
                )
            )

            if active is None:
                return None

            timestamp = max(
                timestamp,
                active.last_activity,
            )

            active.last_activity = timestamp

            if active.speech_end is None:
                active.speech_end = timestamp

            self._session_time[
                session_id
            ] = max(
                self._session_time.get(
                    session_id,
                    0.0,
                ),
                timestamp,
            )

            results = (
                self._finalize_active_segment(
                    session_id,
                    force=force,
                )
            )

            if not results:
                return None

            return results[0]

    # =================================================================
    # FINALIZE
    # =================================================================

    def _finalize_active_segment(
        self,
        session_id: str,
        *,
        force: bool = False,
    ) -> List[SegmentResult]:
        """
        Finalise le segment actif.

        Le segment actif est retiré uniquement lorsque
        sa finalisation est effectivement effectuée.
        """

        active = self._active_segments.get(
            session_id
        )

        if active is None:
            return []

        end = (
            active.speech_end
            if active.speech_end is not None
            else active.last_activity
        )

        end = max(
            active.start,
            end,
        )

        max_end = (
            active.start
            + self.max_segment_duration
        )

        end = min(
            end,
            max_end,
        )

        duration = max(
            0.0,
            end - active.start,
        )

        # ------------------------------------------------------------
        # Segment trop court.
        # ------------------------------------------------------------

        if (
            duration < self.min_segment_duration
            and not force
        ):

            self._active_segments.pop(
                session_id,
                None,
            )

            metrics.increment(
                "segments_rejected"
            )

            return []

        completed = (
            self._completed_segments.setdefault(
                session_id,
                [],
            )
        )

        # ------------------------------------------------------------
        # Limite de segments.
        # ------------------------------------------------------------

        if (
            len(completed)
            >= self.max_segments_per_session
        ):

            self._active_segments.pop(
                session_id,
                None,
            )

            metrics.increment(
                "errors"
            )

            return []

        # ------------------------------------------------------------
        # Le segment est consommé.
        # ------------------------------------------------------------

        self._active_segments.pop(
            session_id,
            None,
        )

        # ------------------------------------------------------------
        # Padding temporel.
        #
        # Aucun silence artificiel n'est ajouté au PCM.
        # ------------------------------------------------------------

        padded_start = max(
            0.0,
            active.start
            - self.segment_padding,
        )

        padded_end = min(
            end + self.segment_padding,
            active.start
            + self.max_segment_duration,
        )

        padded_end = max(
            padded_start,
            padded_end,
        )

        # ------------------------------------------------------------
        # Identifiant.
        # ------------------------------------------------------------

        segment_id = (
            "segment_"
            f"{uuid.uuid4().hex[:12]}"
        )

        audio = bytes(
            active.audio
        )

        confidence = active.confidence

        metadata = {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
            "codec": "pcm16",

            "speech_start": round(
                active.speech_start,
                6,
            ),

            "speech_end": round(
                end,
                6,
            ),

            "padding_ms": (
                self.segment_padding_ms
            ),

            "speech_frames": (
                active.speech_frames
            ),

            "silence_frames": (
                active.silence_frames
            ),

            "audio_duration": round(
                active.audio_duration,
                6,
            ),

            "audio_size": len(
                audio
            ),

            "confidence_samples": (
                active.confidence_count
            ),

            "forced": bool(
                force
            ),
        }

        result = SegmentResult(
            segment_id=segment_id,
            session_id=session_id,
            start=padded_start,
            end=padded_end,
            audio=audio,
            confidence=confidence,
            sequence=active.sequence,
            metadata=metadata,
        )

        completed.append(
            result
        )

        metrics.increment(
            "segments_created"
        )

        return [result]

    # =================================================================
    # FLUSH
    # =================================================================

    def flush_session(
        self,
        session_id: str,
        timestamp: Optional[float] = None,
    ) -> List[SegmentResult]:
        """
        Force la finalisation du segment actif.
        """

        session_id = self._normalize_session_id(
            session_id
        )

        with self._lock:

            active = (
                self._active_segments.get(
                    session_id
                )
            )

            if active is None:
                return []

            if timestamp is None:
                timestamp = active.last_activity

            try:
                timestamp = float(
                    timestamp
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    "timestamp must be numeric."
                ) from exc

            timestamp = max(
                0.0,
                timestamp,
            )

            timestamp = max(
                timestamp,
                active.last_activity,
            )

            active.last_activity = timestamp

            if active.speech_end is None:
                active.speech_end = timestamp

            self._session_time[
                session_id
            ] = max(
                self._session_time.get(
                    session_id,
                    0.0,
                ),
                timestamp,
            )

            return self._finalize_active_segment(
                session_id,
                force=True,
            )

    # =================================================================
    # ALIAS FLUSH
    # =================================================================

    def flush(
        self,
        session_id: Optional[str] = None,
    ) -> List[SegmentResult]:
        """
        Finalise une session ou toutes les sessions.
        """

        if session_id is None:

            results: List[
                SegmentResult
            ] = []

            with self._lock:

                session_ids = list(
                    self._active_segments.keys()
                )

            for current_session_id in session_ids:

                results.extend(
                    self.flush_session(
                        current_session_id
                    )
                )

            return results

        return self.flush_session(
            session_id
        )

    # =================================================================
    # GET ACTIVE
    # =================================================================

    def get_active_segment(
        self,
        session_id: str,
    ) -> Optional[ActiveSegment]:
        """
        Retourne une copie indépendante du segment actif.
        """

        session_id = self._normalize_session_id(
            session_id
        )

        with self._lock:

            active = (
                self._active_segments.get(
                    session_id
                )
            )

            if active is None:
                return None

            return ActiveSegment(
                session_id=active.session_id,
                start=active.start,
                last_activity=active.last_activity,
                audio=bytearray(
                    active.audio
                ),
                speech_start=active.speech_start,
                speech_end=active.speech_end,
                sequence=active.sequence,
                created_at=active.created_at,
                last_was_speech=active.last_was_speech,
                speech_frames=active.speech_frames,
                silence_frames=active.silence_frames,
                audio_duration=active.audio_duration,
                confidence_sum=active.confidence_sum,
                confidence_count=active.confidence_count,
            )

    # =================================================================
    # COMPLETED
    # =================================================================

    def get_segments(
        self,
        session_id: str,
    ) -> List[SegmentResult]:
        """
        Retourne une copie de la liste des segments terminés.
        """

        session_id = self._normalize_session_id(
            session_id
        )

        with self._lock:

            segments = (
                self._completed_segments.get(
                    session_id,
                    [],
                )
            )

            return list(
                segments
            )

    def pop_segments(
        self,
        session_id: str,
    ) -> List[SegmentResult]:
        """
        Retourne puis supprime les segments terminés.
        """

        session_id = self._normalize_session_id(
            session_id
        )

        with self._lock:

            segments = list(
                self._completed_segments.get(
                    session_id,
                    [],
                )
            )

            self._completed_segments[
                session_id
            ] = []

            return segments

    # =================================================================
    # COUNT
    # =================================================================

    def segment_count(
        self,
        session_id: str,
    ) -> int:
        """
        Retourne le nombre de segments terminés.
        """

        session_id = self._normalize_session_id(
            session_id
        )

        with self._lock:

            return len(
                self._completed_segments.get(
                    session_id,
                    [],
                )
            )

    # =================================================================
    # RESET
    # =================================================================

    def reset_session(
        self,
        session_id: str,
    ) -> None:
        """
        Réinitialise complètement une session
        sans supprimer sa structure logique.
        """

        session_id = self._normalize_session_id(
            session_id
        )

        with self._lock:

            self._active_segments.pop(
                session_id,
                None,
            )

            self._completed_segments[
                session_id
            ] = []

            self._sequence[
                session_id
            ] = 0

            self._session_time[
                session_id
            ] = 0.0

    # =================================================================
    # SESSION EXISTS
    # =================================================================

    def has_session(
        self,
        session_id: str,
    ) -> bool:
        """
        Vérifie l'existence logique d'une session.
        """

        session_id = self._normalize_session_id(
            session_id
        )

        with self._lock:

            return (
                session_id
                in self._completed_segments
            )

    # =================================================================
    # ACTIVE
    # =================================================================

    def is_active(
        self,
        session_id: str,
    ) -> bool:
        """
        Vérifie si une session possède un segment actif.
        """

        session_id = self._normalize_session_id(
            session_id
        )

        with self._lock:

            return (
                session_id
                in self._active_segments
            )

    # =================================================================
    # HEALTH
    # =================================================================

    def health(self) -> Dict[str, Any]:
        """
        Retourne l'état de santé du SegmentManager.
        """

        with self._lock:

            active_sessions = len(
                self._active_segments
            )

            completed_sessions = len(
                self._completed_segments
            )

            total_segments = sum(
                len(segments)
                for segments
                in self._completed_segments.values()
            )

            active_audio_bytes = sum(
                len(segment.audio)
                for segment
                in self._active_segments.values()
            )

        return {
            "status": "healthy",
            "component": "segment_manager",

            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,

            "active_sessions": active_sessions,
            "completed_sessions": completed_sessions,
            "total_segments": total_segments,
            "active_audio_bytes": active_audio_bytes,

            "min_segment_duration_ms": (
                self.min_segment_duration_ms
            ),

            "max_segment_duration_ms": (
                self.max_segment_duration_ms
            ),

            "segment_padding_ms": (
                self.segment_padding_ms
            ),

            "max_segments_per_session": (
                self.max_segments_per_session
            ),

            "config_loaded": (
                self.config is not None
            ),
        }

    # =================================================================
    # STATISTICS
    # =================================================================

    def statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques détaillées.
        """

        with self._lock:

            sessions: Dict[
                str,
                Dict[str, Any],
            ] = {}

            session_ids = (
                set(
                    self._completed_segments
                )
                | set(
                    self._active_segments
                )
            )

            for session_id in session_ids:

                segments = (
                    self._completed_segments.get(
                        session_id,
                        [],
                    )
                )

                active = (
                    self._active_segments.get(
                        session_id
                    )
                )

                sessions[
                    session_id
                ] = {
                    "segments": len(
                        segments
                    ),

                    "active": (
                        active is not None
                    ),

                    "sequence": (
                        self._sequence.get(
                            session_id,
                            0,
                        )
                    ),

                    "session_time": round(
                        self._session_time.get(
                            session_id,
                            0.0,
                        ),
                        3,
                    ),

                    "active_duration": (
                        round(
                            active.duration,
                            3,
                        )
                        if active is not None
                        else 0.0
                    ),

                    "active_audio_size": (
                        len(active.audio)
                        if active is not None
                        else 0
                    ),

                    "speech_frames": (
                        active.speech_frames
                        if active is not None
                        else 0
                    ),

                    "silence_frames": (
                        active.silence_frames
                        if active is not None
                        else 0
                    ),

                    "active_confidence": (
                        round(
                            active.confidence,
                            4,
                        )
                        if active is not None
                        else 0.0
                    ),
                }

            return {
                "active_sessions": len(
                    self._active_segments
                ),

                "completed_sessions": len(
                    self._completed_segments
                ),

                "total_segments": sum(
                    len(segments)
                    for segments
                    in self._completed_segments.values()
                ),

                "sessions": sessions,
            }


# ====================================================================
# FACTORY
# ====================================================================

def create_segment_manager(
    config: Optional[Any] = None,
    **overrides: Any,
) -> SegmentManager:
    """
    Factory du SegmentManager.
    """

    return SegmentManager(
        config=config,
        **overrides,
    )


# ====================================================================
# PUBLIC API
# ====================================================================

__all__ = [
    "ActiveSegment",
    "SegmentResult",
    "SegmentManager",
    "create_segment_manager",
]