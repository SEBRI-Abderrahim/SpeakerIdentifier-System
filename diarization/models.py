"""
====================================================================
SpeakerIdentity IA - Diarization Models
====================================================================

Projet       : SpeakerIdentity IA
Module       : Diarization
Fichier      : diarization/models.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 2.3.0
Date         : 14/08/2026

Description
------------

Contrats de données communs au module de diarisation SpeakerIdentity IA.

Cette partie contient :

    - constantes audio
    - helpers de validation
    - énumérations
    - AudioFormat
    - AudioFrame
    - VADResult
    - SpeakerEmbedding
    - SpeakerProfile
    - SpeakerSegment
    - DiarizationResult
    - DiarizationSession
    - SessionInfo

Contraintes audio internes SpeakerIdentity :

    Codec       : PCM16
    Sample rate : 16000 Hz
    Channels    : 1
    Width       : 2 bytes

====================================================================
"""

from __future__ import annotations

import math
import time
import uuid

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


# ====================================================================
# CONSTANTES
# ====================================================================

SpeakerIdentity_SAMPLE_RATE = 16000
SpeakerIdentity_CHANNELS = 1
SpeakerIdentity_SAMPLE_WIDTH = 2

VALID_FRAME_DURATIONS_MS = (10, 20, 30)


# ====================================================================
# HELPERS INTERNES
# ====================================================================

def _validate_finite(
    value: float,
    field_name: str,
) -> float:
    """Vérifie qu'une valeur numérique est finie."""

    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{field_name} must be a numeric value."
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            f"{field_name} must be a finite number."
        )

    return value


def _validate_non_negative(
    value: float,
    field_name: str,
) -> float:
    """Valide une valeur numérique >= 0."""

    value = _validate_finite(
        value,
        field_name,
    )

    if value < 0:
        raise ValueError(
            f"{field_name} must be >= 0."
        )

    return value


def _validate_non_negative_int(
    value: int,
    field_name: str,
) -> int:
    """Valide un entier >= 0."""

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


def _validate_positive_int(
    value: int,
    field_name: str,
) -> int:
    """Valide un entier strictement positif."""

    if isinstance(value, bool):
        raise TypeError(
            f"{field_name} must be an integer."
        )

    if not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{field_name} must be positive."
        )

    return value


def _validate_bool(
    value: bool,
    field_name: str,
) -> bool:
    """Valide explicitement une valeur booléenne."""

    if not isinstance(value, bool):
        raise TypeError(
            f"{field_name} must be a boolean."
        )

    return value


def _clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """Limite une valeur dans un intervalle."""

    value = _validate_finite(
        value,
        "value",
    )

    minimum = _validate_finite(
        minimum,
        "minimum",
    )

    maximum = _validate_finite(
        maximum,
        "maximum",
    )

    if minimum > maximum:
        raise ValueError(
            "minimum cannot be greater than maximum."
        )

    return max(
        minimum,
        min(maximum, value),
    )


def _normalize_id(
    value: str,
    field_name: str,
) -> str:
    """Normalise et valide un identifiant."""

    if value is None:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    value = str(value).strip()

    if not value:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    return value


def _normalize_event_type(
    value: "DiarizationEventType | str",
) -> "DiarizationEventType | str":
    """
    Normalise un event_type vers l'Enum lorsqu'il
    correspond à une valeur connue.
    """

    if isinstance(
        value,
        DiarizationEventType,
    ):
        return value

    if value is None:
        raise ValueError(
            "event_type cannot be empty."
        )

    value = str(value).strip()

    if not value:
        raise ValueError(
            "event_type cannot be empty."
        )

    try:
        return DiarizationEventType(value)
    except ValueError:
        return value


# ====================================================================
# ENUMERATIONS
# ====================================================================

class AudioCodec(str, Enum):
    """Codec audio interne SpeakerIdentity."""

    PCM16 = "pcm16"


class SessionStatus(str, Enum):
    """État d'une session de diarisation."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class SpeakerState(str, Enum):
    """État logique d'un speaker."""

    UNKNOWN = "unknown"
    ACTIVE = "active"
    INACTIVE = "inactive"


class VADState(str, Enum):
    """État de détection vocale."""

    SPEECH = "speech"
    SILENCE = "silence"


class DiarizationEventType(str, Enum):
    """Événements standards du module."""

    SESSION_STARTED = "session.started"
    SESSION_STOPPED = "session.stopped"

    AUDIO_RECEIVED = "audio.received"

    SPEECH_STARTED = "speech.started"
    SPEECH_ENDED = "speech.ended"

    SEGMENT_CREATED = "segment.created"

    SPEAKER_DETECTED = "speaker.detected"
    SPEAKER_IDENTIFIED = "speaker.identified"
    SPEAKER_CREATED = "speaker.created"
    SPEAKER_CHANGED = "speaker.changed"

    DIARIZATION_RESULT = "diarization.result"

    ERROR = "diarization.error"


# ====================================================================
# AUDIO FORMAT
# ====================================================================

@dataclass(frozen=True)
class AudioFormat:
    """Description du format audio SpeakerIdentity."""

    codec: AudioCodec = AudioCodec.PCM16

    sample_rate: int = SpeakerIdentity_SAMPLE_RATE

    channels: int = SpeakerIdentity_CHANNELS

    sample_width: int = SpeakerIdentity_SAMPLE_WIDTH

    frame_duration_ms: int = 30

    def __post_init__(self) -> None:

        codec = self.codec

        if not isinstance(codec, AudioCodec):
            try:
                codec = AudioCodec(codec)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Unsupported audio codec."
                ) from exc

            object.__setattr__(
                self,
                "codec",
                codec,
            )

        _validate_positive_int(
            self.sample_rate,
            "sample_rate",
        )

        if self.sample_rate != SpeakerIdentity_SAMPLE_RATE:
            raise ValueError(
                "SpeakerIdentity diarization requires "
                f"{SpeakerIdentity_SAMPLE_RATE} Hz audio."
            )

        _validate_positive_int(
            self.channels,
            "channels",
        )

        if self.channels != SpeakerIdentity_CHANNELS:
            raise ValueError(
                "SpeakerIdentity diarization requires mono audio."
            )

        _validate_positive_int(
            self.sample_width,
            "sample_width",
        )

        if self.sample_width != SpeakerIdentity_SAMPLE_WIDTH:
            raise ValueError(
                "SpeakerIdentity diarization requires PCM16 "
                "(sample_width=2)."
            )

        if isinstance(
            self.frame_duration_ms,
            bool,
        ):
            raise TypeError(
                "frame_duration_ms must be an integer."
            )

        if not isinstance(
            self.frame_duration_ms,
            int,
        ):
            raise TypeError(
                "frame_duration_ms must be an integer."
            )

        if (
            self.frame_duration_ms
            not in VALID_FRAME_DURATIONS_MS
        ):
            raise ValueError(
                "frame_duration_ms must be 10, 20 or 30."
            )

    @property
    def bytes_per_sample(self) -> int:
        """Nombre d'octets par échantillon et par canal."""

        return self.sample_width

    @property
    def samples_per_frame(self) -> int:
        """Nombre d'échantillons dans une frame."""

        return int(
            self.sample_rate
            * self.frame_duration_ms
            / 1000
        )

    @property
    def bytes_per_frame(self) -> int:
        """Nombre total d'octets d'une frame."""

        return (
            self.samples_per_frame
            * self.channels
            * self.sample_width
        )

    @property
    def frame_duration_seconds(self) -> float:
        """Durée d'une frame en secondes."""

        return self.frame_duration_ms / 1000.0

    @property
    def frame_duration(self) -> float:
        """Alias historique."""

        return self.frame_duration_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise le format audio."""

        return {
            "codec": self.codec.value,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
            "frame_duration_ms": self.frame_duration_ms,
            "frame_duration_seconds": (
                self.frame_duration_seconds
            ),
            "samples_per_frame": (
                self.samples_per_frame
            ),
            "bytes_per_frame": (
                self.bytes_per_frame
            ),
        }


# ====================================================================
# AUDIO FRAME
# ====================================================================

@dataclass
class AudioFrame:
    """Frame PCM16 utilisée par le pipeline temps réel."""

    data: bytes

    timestamp: float

    duration: float

    sample_rate: int = SpeakerIdentity_SAMPLE_RATE

    channels: int = SpeakerIdentity_CHANNELS

    sample_width: int = SpeakerIdentity_SAMPLE_WIDTH

    session_id: Optional[str] = None

    sequence_number: int = 0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        if not isinstance(
            self.data,
            bytes,
        ):
            raise TypeError(
                "AudioFrame.data must be bytes."
            )

        _validate_positive_int(
            self.sample_rate,
            "AudioFrame.sample_rate",
        )

        if self.sample_rate != SpeakerIdentity_SAMPLE_RATE:
            raise ValueError(
                "AudioFrame.sample_rate must be "
                f"{SpeakerIdentity_SAMPLE_RATE} Hz."
            )

        _validate_positive_int(
            self.channels,
            "AudioFrame.channels",
        )

        if self.channels != SpeakerIdentity_CHANNELS:
            raise ValueError(
                "AudioFrame requires mono audio."
            )

        _validate_positive_int(
            self.sample_width,
            "AudioFrame.sample_width",
        )

        if self.sample_width != SpeakerIdentity_SAMPLE_WIDTH:
            raise ValueError(
                "AudioFrame requires PCM16."
            )

        self.timestamp = _validate_non_negative(
            self.timestamp,
            "AudioFrame.timestamp",
        )

        self.duration = _validate_non_negative(
            self.duration,
            "AudioFrame.duration",
        )

        self.sequence_number = _validate_non_negative_int(
            self.sequence_number,
            "AudioFrame.sequence_number",
        )

        if self.session_id is not None:
            self.session_id = _normalize_id(
                self.session_id,
                "session_id",
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise TypeError(
                "AudioFrame.metadata must be a dictionary."
            )

        bytes_per_sample = (
            self.channels
            * self.sample_width
        )

        if (
            len(self.data)
            % bytes_per_sample
            != 0
        ):
            raise ValueError(
                "AudioFrame.data size must be aligned "
                "to complete PCM samples."
            )

        expected_duration = (
            len(self.data)
            / (
                self.sample_rate
                * self.channels
                * self.sample_width
            )
        )

        if not self.data:
            if self.duration != 0:
                raise ValueError(
                    "AudioFrame.duration must be 0 "
                    "when data is empty."
                )

            return

        if self.duration <= 0:
            raise ValueError(
                "AudioFrame.duration must be > 0 "
                "when data is not empty."
            )

        tolerance = max(
            0.001,
            expected_duration * 0.10,
        )

        if abs(
            self.duration - expected_duration
        ) > tolerance:
            raise ValueError(
                "AudioFrame.duration is inconsistent "
                "with PCM data. Expected approximately "
                f"{expected_duration:.6f}s, got "
                f"{self.duration:.6f}s."
            )

    @property
    def size(self) -> int:
        """Taille de la frame en octets."""

        return len(self.data)

    @property
    def samples(self) -> int:
        """Nombre d'échantillons PCM."""

        bytes_per_sample = (
            self.channels
            * self.sample_width
        )

        if bytes_per_sample <= 0:
            return 0

        return (
            len(self.data)
            // bytes_per_sample
        )

    @property
    def end_timestamp(self) -> float:
        """Timestamp de fin."""

        return self.timestamp + self.duration

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise la frame sans les données PCM."""

        return {
            "timestamp": self.timestamp,
            "duration": self.duration,
            "end_timestamp": self.end_timestamp,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
            "session_id": self.session_id,
            "sequence_number": self.sequence_number,
            "size": self.size,
            "samples": self.samples,
            "metadata": dict(self.metadata),
        }


# ====================================================================
# VAD RESULT
# ====================================================================

@dataclass
class VADResult:
    """Résultat du traitement VAD."""

    is_speech: bool

    timestamp: float

    duration: float

    confidence: float = 1.0

    backend: str = "webrtc"

    session_id: Optional[str] = None

    sequence_number: int = 0

    rms: Optional[float] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        self.is_speech = _validate_bool(
            self.is_speech,
            "VADResult.is_speech",
        )

        self.timestamp = _validate_non_negative(
            self.timestamp,
            "VADResult.timestamp",
        )

        self.duration = _validate_non_negative(
            self.duration,
            "VADResult.duration",
        )

        self.confidence = _clamp(
            self.confidence,
            0.0,
            1.0,
        )

        if self.backend is None:
            raise ValueError(
                "VADResult.backend cannot be empty."
            )

        self.backend = str(
            self.backend
        ).strip().lower()

        if not self.backend:
            raise ValueError(
                "VADResult.backend cannot be empty."
            )

        self.sequence_number = _validate_non_negative_int(
            self.sequence_number,
            "VADResult.sequence_number",
        )

        if self.session_id is not None:
            self.session_id = _normalize_id(
                self.session_id,
                "session_id",
            )

        if self.rms is not None:
            self.rms = _validate_non_negative(
                self.rms,
                "VADResult.rms",
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise TypeError(
                "VADResult.metadata must be a dictionary."
            )

    @property
    def state(self) -> VADState:
        """Retourne l'état logique du VAD."""

        return (
            VADState.SPEECH
            if self.is_speech
            else VADState.SILENCE
        )

    @property
    def end_timestamp(self) -> float:
        """Timestamp de fin."""

        return self.timestamp + self.duration

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise le résultat VAD."""

        return {
            "is_speech": self.is_speech,
            "state": self.state.value,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "end_timestamp": self.end_timestamp,
            "confidence": self.confidence,
            "backend": self.backend,
            "session_id": self.session_id,
            "sequence_number": self.sequence_number,
            "rms": self.rms,
            "metadata": dict(self.metadata),
        }


# ====================================================================
# SPEAKER EMBEDDING
# ====================================================================

@dataclass
class SpeakerEmbedding:
    """Embedding vocal associé à un speaker."""

    vector: Sequence[float]

    dimension: int

    model: str = ""

    session_id: Optional[str] = None

    speaker_id: Optional[str] = None

    timestamp: float = field(
        default_factory=time.time
    )

    confidence: float = 1.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        self.dimension = _validate_positive_int(
            self.dimension,
            "Embedding dimension",
        )

        if self.vector is None:
            raise TypeError(
                "Embedding vector cannot be None."
            )

        try:
            vector_length = len(self.vector)
        except TypeError as exc:
            raise TypeError(
                "Embedding vector must be a sized sequence."
            ) from exc

        if vector_length != self.dimension:
            raise ValueError(
                "Embedding vector length "
                f"({vector_length}) does not match "
                f"dimension ({self.dimension})."
            )

        self.timestamp = _validate_non_negative(
            self.timestamp,
            "Embedding timestamp",
        )

        self.confidence = _clamp(
            self.confidence,
            0.0,
            1.0,
        )

        if self.model is None:
            self.model = ""

        self.model = str(
            self.model
        ).strip()

        if self.session_id is not None:
            self.session_id = _normalize_id(
                self.session_id,
                "session_id",
            )

        if self.speaker_id is not None:
            self.speaker_id = _normalize_id(
                self.speaker_id,
                "speaker_id",
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise TypeError(
                "Embedding metadata must be a dictionary."
            )

        for value in self.vector:
            _validate_finite(
                value,
                "Embedding vector value",
            )

    def to_list(self) -> List[float]:
        """Convertit le vecteur en liste Python."""

        return [
            float(value)
            for value in self.vector
        ]

    def to_dict(
        self,
        include_vector: bool = False,
    ) -> Dict[str, Any]:
        """Sérialise l'embedding."""

        result: Dict[str, Any] = {
            "dimension": self.dimension,
            "model": self.model,
            "session_id": self.session_id,
            "speaker_id": self.speaker_id,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }

        if include_vector:
            result["vector"] = self.to_list()

        return result


# ====================================================================
# SPEAKER PROFILE
# ====================================================================

@dataclass
class SpeakerProfile:
    """Profil logique d'un speaker."""

    speaker_id: str

    session_id: str

    state: SpeakerState = SpeakerState.UNKNOWN

    sample_count: int = 0

    segment_count: int = 0

    total_speech_duration: float = 0.0

    created_at: float = field(
        default_factory=time.time
    )

    last_seen: float = field(
        default_factory=time.time
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        self.speaker_id = _normalize_id(
            self.speaker_id,
            "speaker_id",
        )

        self.session_id = _normalize_id(
            self.session_id,
            "session_id",
        )

        if not isinstance(
            self.state,
            SpeakerState,
        ):
            try:
                self.state = SpeakerState(
                    self.state
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid speaker state."
                ) from exc

        self.sample_count = _validate_non_negative_int(
            self.sample_count,
            "sample_count",
        )

        self.segment_count = _validate_non_negative_int(
            self.segment_count,
            "segment_count",
        )

        self.total_speech_duration = _validate_non_negative(
            self.total_speech_duration,
            "total_speech_duration",
        )

        self.created_at = _validate_non_negative(
            self.created_at,
            "created_at",
        )

        self.last_seen = _validate_non_negative(
            self.last_seen,
            "last_seen",
        )

        if self.last_seen < self.created_at:
            raise ValueError(
                "last_seen cannot be earlier than created_at."
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise TypeError(
                "SpeakerProfile.metadata must be a dictionary."
            )

    def touch(
        self,
        timestamp: Optional[float] = None,
    ) -> None:
        """Met à jour la dernière activité."""

        if timestamp is None:
            timestamp = time.time()

        timestamp = _validate_non_negative(
            timestamp,
            "timestamp",
        )

        if timestamp < self.created_at:
            raise ValueError(
                "timestamp cannot be earlier than created_at."
            )

        self.last_seen = timestamp
        self.state = SpeakerState.ACTIVE

    def mark_inactive(self) -> None:
        """Marque le speaker comme inactif."""

        self.state = SpeakerState.INACTIVE

    def add_segment(
        self,
        duration: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Enregistre un segment vocal."""

        duration = _validate_non_negative(
            duration,
            "duration",
        )

        self.segment_count += 1
        self.sample_count += 1
        self.total_speech_duration += duration

        self.touch(timestamp)

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise le profil."""

        return {
            "speaker_id": self.speaker_id,
            "session_id": self.session_id,
            "state": self.state.value,
            "sample_count": self.sample_count,
            "segment_count": self.segment_count,
            "total_speech_duration": round(
                self.total_speech_duration,
                3,
            ),
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            "metadata": dict(self.metadata),
        }


# ====================================================================
# SPEAKER SEGMENT
# ====================================================================

@dataclass
class SpeakerSegment:
    """Segment vocal attribué à un speaker."""

    session_id: str

    speaker_id: str

    start: float

    end: float

    confidence: float = 1.0

    segment_id: str = field(
        default_factory=lambda: (
            f"segment_{uuid.uuid4().hex[:12]}"
        )
    )

    audio_offset: Optional[int] = None

    audio_size: Optional[int] = None

    transcription: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    created_at: float = field(
        default_factory=time.time
    )

    def __post_init__(self) -> None:

        self.session_id = _normalize_id(
            self.session_id,
            "session_id",
        )

        self.speaker_id = _normalize_id(
            self.speaker_id,
            "speaker_id",
        )

        self.segment_id = _normalize_id(
            self.segment_id,
            "segment_id",
        )

        self.start = _validate_non_negative(
            self.start,
            "Segment start",
        )

        self.end = _validate_non_negative(
            self.end,
            "Segment end",
        )

        if self.end < self.start:
            raise ValueError(
                "Segment end must be >= start."
            )

        self.confidence = _clamp(
            self.confidence,
            0.0,
            1.0,
        )

        if self.audio_offset is not None:
            self.audio_offset = _validate_non_negative_int(
                self.audio_offset,
                "audio_offset",
            )

        if self.audio_size is not None:
            self.audio_size = _validate_non_negative_int(
                self.audio_size,
                "audio_size",
            )

        if (
            self.audio_offset is None
            and self.audio_size is not None
        ):
            raise ValueError(
                "audio_size cannot be defined without "
                "audio_offset."
            )

        self.created_at = _validate_non_negative(
            self.created_at,
            "created_at",
        )

        if self.transcription is not None:
            self.transcription = str(
                self.transcription
            ).strip()

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise TypeError(
                "SpeakerSegment.metadata must be a dictionary."
            )

    @property
    def duration(self) -> float:
        """Durée du segment."""

        return self.end - self.start

    def set_transcription(
        self,
        text: Optional[str],
    ) -> None:
        """Associe une transcription STT."""

        if text is None:
            self.transcription = None
            return

        self.transcription = str(
            text
        ).strip()

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise le segment."""

        return {
            "segment_id": self.segment_id,
            "session_id": self.session_id,
            "speaker_id": self.speaker_id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "confidence": round(
                self.confidence,
                4,
            ),
            "audio_offset": self.audio_offset,
            "audio_size": self.audio_size,
            "transcription": self.transcription,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


# ====================================================================
# DIARIZATION RESULT
# ====================================================================

@dataclass
class DiarizationResult:
    """Résultat produit par le pipeline de diarisation."""

    session_id: str

    segments: List[SpeakerSegment] = field(
        default_factory=list
    )

    current_speaker_id: Optional[str] = None

    processing_time: float = 0.0

    audio_duration: float = 0.0

    speech_duration: float = 0.0

    confidence: float = 0.0

    completed: bool = False

    timestamp: float = field(
        default_factory=time.time
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        self.session_id = _normalize_id(
            self.session_id,
            "session_id",
        )

        if not isinstance(
            self.segments,
            list,
        ):
            raise TypeError(
                "segments must be a list."
            )

        self.processing_time = _validate_non_negative(
            self.processing_time,
            "processing_time",
        )

        self.audio_duration = _validate_non_negative(
            self.audio_duration,
            "audio_duration",
        )

        self.speech_duration = _validate_non_negative(
            self.speech_duration,
            "speech_duration",
        )

        if self.speech_duration > self.audio_duration:
            raise ValueError(
                "speech_duration cannot be greater "
                "than audio_duration."
            )

        self.confidence = _clamp(
            self.confidence,
            0.0,
            1.0,
        )

        self.completed = _validate_bool(
            self.completed,
            "completed",
        )

        self.timestamp = _validate_non_negative(
            self.timestamp,
            "timestamp",
        )

        if self.current_speaker_id is not None:
            self.current_speaker_id = _normalize_id(
                self.current_speaker_id,
                "current_speaker_id",
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise TypeError(
                "DiarizationResult.metadata must be a dictionary."
            )

        calculated_speech_duration = 0.0

        for segment in self.segments:

            if not isinstance(
                segment,
                SpeakerSegment,
            ):
                raise TypeError(
                    "All segments must be "
                    "SpeakerSegment instances."
                )

            if segment.session_id != self.session_id:
                raise ValueError(
                    "All segments must belong to "
                    "the same session."
                )

            if segment.end > self.audio_duration:
                raise ValueError(
                    "Segment end cannot be greater "
                    "than audio_duration."
                )

            calculated_speech_duration += (
                segment.duration
            )

        if calculated_speech_duration > self.audio_duration:
            raise ValueError(
                "Total segment duration cannot be "
                "greater than audio_duration."
            )

        if self.segments:

            tolerance = max(
                0.001,
                self.audio_duration * 0.01,
            )

            if abs(
                self.speech_duration
                - calculated_speech_duration
            ) > tolerance:
                raise ValueError(
                    "speech_duration is inconsistent "
                    "with the supplied segments. "
                    f"Expected approximately "
                    f"{calculated_speech_duration:.6f}s, "
                    f"got {self.speech_duration:.6f}s."
                )

    @property
    def speaker_ids(self) -> List[str]:
        """Retourne les speakers uniques."""

        result: List[str] = []
        seen = set()

        for segment in self.segments:

            speaker_id = segment.speaker_id

            if speaker_id not in seen:
                seen.add(speaker_id)
                result.append(speaker_id)

        return result

    @property
    def speaker_count(self) -> int:
        """Nombre de speakers."""

        return len(self.speaker_ids)

    @property
    def segment_count(self) -> int:
        """Nombre de segments."""

        return len(self.segments)

    @property
    def speech_ratio(self) -> float:
        """Ratio parole/audio."""

        if self.audio_duration <= 0:
            return 0.0

        return _clamp(
            self.speech_duration
            / self.audio_duration,
            0.0,
            1.0,
        )

    def add_segment(
        self,
        segment: SpeakerSegment,
    ) -> None:
        """Ajoute un segment."""

        if not isinstance(
            segment,
            SpeakerSegment,
        ):
            raise TypeError(
                "segment must be a SpeakerSegment instance."
            )

        if segment.session_id != self.session_id:
            raise ValueError(
                "Segment session_id does not match "
                "DiarizationResult."
            )

        if segment.end > self.audio_duration:
            raise ValueError(
                "Segment end cannot be greater "
                "than audio_duration."
            )

        new_speech_duration = (
            self.speech_duration
            + segment.duration
        )

        if new_speech_duration > self.audio_duration:
            raise ValueError(
                "Adding this segment would make "
                "speech_duration greater than "
                "audio_duration."
            )

        self.segments.append(segment)

        self.speech_duration = new_speech_duration

        self.confidence = (
            sum(
                item.confidence
                for item in self.segments
            )
            / len(self.segments)
        )

        self.current_speaker_id = (
            segment.speaker_id
        )

    def to_dict(
        self,
        include_segments: bool = True,
    ) -> Dict[str, Any]:
        """Sérialise le résultat."""

        result: Dict[str, Any] = {
            "session_id": self.session_id,
            "current_speaker_id": (
                self.current_speaker_id
            ),
            "speaker_ids": self.speaker_ids,
            "speaker_count": self.speaker_count,
            "segment_count": self.segment_count,
            "audio_duration": round(
                self.audio_duration,
                3,
            ),
            "speech_duration": round(
                self.speech_duration,
                3,
            ),
            "speech_ratio": round(
                self.speech_ratio,
                4,
            ),
            "processing_time": round(
                self.processing_time,
                6,
            ),
            "confidence": round(
                self.confidence,
                4,
            ),
            "completed": self.completed,
            "timestamp": self.timestamp,
            "metadata": dict(
                self.metadata
            ),
        }

        if include_segments:
            result["segments"] = [
                segment.to_dict()
                for segment in self.segments
            ]

        return result


# ====================================================================
# DIARIZATION SESSION
# ====================================================================

@dataclass
class DiarizationSession:
    """Contexte complet d'une session de diarisation."""

    session_id: str

    status: SessionStatus = SessionStatus.CREATED

    audio_format: AudioFormat = field(
        default_factory=AudioFormat
    )

    speakers: Dict[str, SpeakerProfile] = field(
        default_factory=dict
    )

    segments: List[SpeakerSegment] = field(
        default_factory=list
    )

    current_speaker_id: Optional[str] = None

    total_audio_seconds: float = 0.0

    total_speech_seconds: float = 0.0

    total_segments: int = 0

    total_speaker_changes: int = 0

    created_at: float = field(
        default_factory=time.time
    )

    started_at: Optional[float] = None

    stopped_at: Optional[float] = None

    last_activity: float = field(
        default_factory=time.time
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Valide et normalise la session."""

        self.session_id = _normalize_id(
            self.session_id,
            "session_id",
        )

        if not isinstance(
            self.status,
            SessionStatus,
        ):
            try:
                self.status = SessionStatus(
                    self.status
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid session status."
                ) from exc

        if not isinstance(
            self.audio_format,
            AudioFormat,
        ):
            raise TypeError(
                "audio_format must be an AudioFormat instance."
            )

        if not isinstance(
            self.speakers,
            dict,
        ):
            raise TypeError(
                "speakers must be a dictionary."
            )

        if not isinstance(
            self.segments,
            list,
        ):
            raise TypeError(
                "segments must be a list."
            )

        self.total_audio_seconds = _validate_non_negative(
            self.total_audio_seconds,
            "total_audio_seconds",
        )

        self.total_speech_seconds = _validate_non_negative(
            self.total_speech_seconds,
            "total_speech_seconds",
        )

        if (
            self.total_speech_seconds
            > self.total_audio_seconds
        ):
            raise ValueError(
                "total_speech_seconds cannot be greater "
                "than total_audio_seconds."
            )

        self.total_segments = _validate_non_negative_int(
            self.total_segments,
            "total_segments",
        )

        self.total_speaker_changes = _validate_non_negative_int(
            self.total_speaker_changes,
            "total_speaker_changes",
        )

        self.created_at = _validate_non_negative(
            self.created_at,
            "created_at",
        )

        self.last_activity = _validate_non_negative(
            self.last_activity,
            "last_activity",
        )

        if self.started_at is not None:
            self.started_at = _validate_non_negative(
                self.started_at,
                "started_at",
            )

        if self.stopped_at is not None:
            self.stopped_at = _validate_non_negative(
                self.stopped_at,
                "stopped_at",
            )

        if self.last_activity < self.created_at:
            raise ValueError(
                "last_activity cannot be earlier "
                "than created_at."
            )

        if (
            self.started_at is not None
            and self.started_at < self.created_at
        ):
            raise ValueError(
                "started_at cannot be earlier "
                "than created_at."
            )

        if (
            self.stopped_at is not None
            and self.started_at is not None
            and self.stopped_at < self.started_at
        ):
            raise ValueError(
                "stopped_at cannot be earlier "
                "than started_at."
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise TypeError(
                "DiarizationSession.metadata "
                "must be a dictionary."
            )

        for speaker_id, profile in self.speakers.items():

            if not isinstance(
                profile,
                SpeakerProfile,
            ):
                raise TypeError(
                    "All speakers must be "
                    "SpeakerProfile instances."
                )

            if profile.session_id != self.session_id:
                raise ValueError(
                    "All speakers must belong to "
                    "the same session."
                )

            if speaker_id != profile.speaker_id:
                raise ValueError(
                    "Speaker dictionary key does not "
                    "match profile.speaker_id."
                )

        calculated_segment_duration = 0.0

        for segment in self.segments:

            if not isinstance(
                segment,
                SpeakerSegment,
            ):
                raise TypeError(
                    "All segments must be "
                    "SpeakerSegment instances."
                )

            if segment.session_id != self.session_id:
                raise ValueError(
                    "All segments must belong to "
                    "the same session."
                )

            if segment.speaker_id not in self.speakers:
                raise ValueError(
                    "Segment speaker_id does not "
                    "belong to the session."
                )

            if segment.end > self.total_audio_seconds:
                raise ValueError(
                    "Segment end cannot be greater "
                    "than total_audio_seconds."
                )

            calculated_segment_duration += (
                segment.duration
            )

        if calculated_segment_duration > self.total_audio_seconds:
            raise ValueError(
                "Total segment duration cannot be "
                "greater than total_audio_seconds."
            )

        self.total_segments = len(
            self.segments
        )

    # ----------------------------------------------------------------
    # SESSION STATE
    # ----------------------------------------------------------------

    def start(self) -> None:
        """Démarre la session."""

        now = time.time()

        if self.status == SessionStatus.STOPPED:
            raise ValueError(
                "A stopped session cannot be restarted."
            )

        self.status = SessionStatus.RUNNING

        if self.started_at is None:
            self.started_at = now

        self.stopped_at = None
        self.last_activity = now

    def pause(self) -> None:
        """Met la session en pause."""

        if self.status == SessionStatus.STOPPED:
            raise ValueError(
                "A stopped session cannot be paused."
            )

        self.status = SessionStatus.PAUSED
        self.last_activity = time.time()

    def stop(self) -> None:
        """Arrête la session."""

        now = time.time()

        if (
            self.started_at is not None
            and now < self.started_at
        ):
            now = self.started_at

        self.status = SessionStatus.STOPPED
        self.stopped_at = now
        self.last_activity = now

    def set_error(self) -> None:
        """Place la session en erreur."""

        self.status = SessionStatus.ERROR
        self.last_activity = time.time()

    # ----------------------------------------------------------------
    # SPEAKERS
    # ----------------------------------------------------------------

    def add_speaker(
        self,
        profile: SpeakerProfile,
    ) -> None:
        """Ajoute ou remplace un speaker."""

        if not isinstance(
            profile,
            SpeakerProfile,
        ):
            raise TypeError(
                "profile must be a SpeakerProfile instance."
            )

        if profile.session_id != self.session_id:
            raise ValueError(
                "Speaker session_id does not match "
                "DiarizationSession."
            )

        self.speakers[
            profile.speaker_id
        ] = profile

        self.last_activity = time.time()

    # ----------------------------------------------------------------
    # SEGMENTS
    # ----------------------------------------------------------------

    def add_segment(
        self,
        segment: SpeakerSegment,
    ) -> None:
        """Ajoute un segment à la session."""

        if not isinstance(
            segment,
            SpeakerSegment,
        ):
            raise TypeError(
                "segment must be a SpeakerSegment instance."
            )

        if segment.session_id != self.session_id:
            raise ValueError(
                "Segment session_id does not match "
                "DiarizationSession."
            )

        if segment.speaker_id not in self.speakers:
            raise ValueError(
                "Segment speaker_id does not belong "
                "to the session."
            )

        if segment.end > self.total_audio_seconds:
            raise ValueError(
                "Segment end cannot be greater "
                "than total_audio_seconds."
            )

        self.segments.append(segment)

        self.total_segments = len(
            self.segments
        )

        self.total_speech_seconds += (
            segment.duration
        )

        self.total_speech_seconds = min(
            self.total_speech_seconds,
            self.total_audio_seconds,
        )

        self.set_current_speaker(
            segment.speaker_id
        )

        self.last_activity = time.time()

    # ----------------------------------------------------------------
    # CURRENT SPEAKER
    # ----------------------------------------------------------------

    def set_current_speaker(
        self,
        speaker_id: Optional[str],
    ) -> bool:
        """
        Définit le speaker courant.

        Retourne True lorsqu'un changement réel
        de speaker est détecté.
        """

        if speaker_id is not None:

            speaker_id = _normalize_id(
                speaker_id,
                "speaker_id",
            )

            if speaker_id not in self.speakers:
                raise ValueError(
                    f"Unknown speaker_id: {speaker_id}"
                )

        previous = self.current_speaker_id

        self.current_speaker_id = speaker_id

        changed = (
            previous is not None
            and speaker_id is not None
            and previous != speaker_id
        )

        if changed:
            self.total_speaker_changes += 1

        self.last_activity = time.time()

        return changed

    # ----------------------------------------------------------------
    # AUDIO STATISTICS
    # ----------------------------------------------------------------

    def add_audio_time(
        self,
        duration: float,
        speech: bool = False,
    ) -> None:
        """
        Comptabilise le temps audio reçu.

        total_audio_seconds constitue la source
        de vérité de la durée audio.
        """

        duration = _validate_non_negative(
            duration,
            "duration",
        )

        speech = _validate_bool(
            speech,
            "speech",
        )

        self.total_audio_seconds += duration

        if speech:
            self.total_speech_seconds += duration

        self.total_speech_seconds = min(
            self.total_speech_seconds,
            self.total_audio_seconds,
        )

        self.last_activity = time.time()

    def add_speech_time(
        self,
        duration: float,
    ) -> None:
        """Ajoute uniquement du temps de parole."""

        duration = _validate_non_negative(
            duration,
            "duration",
        )

        new_speech_seconds = (
            self.total_speech_seconds
            + duration
        )

        if new_speech_seconds > self.total_audio_seconds:
            raise ValueError(
                "speech duration cannot be greater "
                "than total audio duration."
            )

        self.total_speech_seconds = (
            new_speech_seconds
        )

        self.last_activity = time.time()

    # ----------------------------------------------------------------
    # PROPERTIES
    # ----------------------------------------------------------------

    @property
    def speaker_count(self) -> int:
        """Nombre de speakers."""

        return len(self.speakers)

    @property
    def duration(self) -> float:
        """Durée audio totale."""

        return self.total_audio_seconds

    @property
    def speech_ratio(self) -> float:
        """Ratio parole/audio."""

        if self.total_audio_seconds <= 0:
            return 0.0

        return _clamp(
            self.total_speech_seconds
            / self.total_audio_seconds,
            0.0,
            1.0,
        )

    # ----------------------------------------------------------------
    # SERIALIZATION
    # ----------------------------------------------------------------

    def to_dict(
        self,
        include_segments: bool = True,
    ) -> Dict[str, Any]:
        """Sérialise la session."""

        result: Dict[str, Any] = {
            "session_id": self.session_id,
            "status": self.status.value,
            "audio_format": (
                self.audio_format.to_dict()
            ),
            "speaker_count": self.speaker_count,
            "speakers": [
                speaker.to_dict()
                for speaker in self.speakers.values()
            ],
            "current_speaker_id": (
                self.current_speaker_id
            ),
            "total_audio_seconds": round(
                self.total_audio_seconds,
                3,
            ),
            "total_speech_seconds": round(
                self.total_speech_seconds,
                3,
            ),
            "speech_ratio": round(
                self.speech_ratio,
                4,
            ),
            "total_segments": self.total_segments,
            "total_speaker_changes": (
                self.total_speaker_changes
            ),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "last_activity": self.last_activity,
            "metadata": dict(
                self.metadata
            ),
        }

        if include_segments:
            result["segments"] = [
                segment.to_dict()
                for segment in self.segments
            ]

        return result


# ====================================================================
# SESSION INFO
# ====================================================================

@dataclass
class SessionInfo:
    """Vue légère d'une session."""

    session_id: str

    status: SessionStatus

    speaker_count: int = 0

    segment_count: int = 0

    duration: float = 0.0

    speech_duration: float = 0.0

    created_at: float = field(
        default_factory=time.time
    )

    last_activity: float = field(
        default_factory=time.time
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        self.session_id = _normalize_id(
            self.session_id,
            "session_id",
        )

        if not isinstance(
            self.status,
            SessionStatus,
        ):
            try:
                self.status = SessionStatus(
                    self.status
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid session status."
                ) from exc

        self.speaker_count = _validate_non_negative_int(
            self.speaker_count,
            "speaker_count",
        )

        self.segment_count = _validate_non_negative_int(
            self.segment_count,
            "segment_count",
        )

        self.duration = _validate_non_negative(
            self.duration,
            "duration",
        )

        self.speech_duration = _validate_non_negative(
            self.speech_duration,
            "speech_duration",
        )

        if self.speech_duration > self.duration:
            raise ValueError(
                "speech_duration cannot be greater "
                "than duration."
            )

        self.created_at = _validate_non_negative(
            self.created_at,
            "created_at",
        )

        self.last_activity = _validate_non_negative(
            self.last_activity,
            "last_activity",
        )

        if self.last_activity < self.created_at:
            raise ValueError(
                "last_activity cannot be earlier "
                "than created_at."
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise TypeError(
                "SessionInfo.metadata must be a dictionary."
            )

    @property
    def speech_ratio(self) -> float:
        """Ratio parole/audio."""

        if self.duration <= 0:
            return 0.0

        return _clamp(
            self.speech_duration
            / self.duration,
            0.0,
            1.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise les informations de session."""

        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "speaker_count": self.speaker_count,
            "segment_count": self.segment_count,
            "duration": round(
                self.duration,
                3,
            ),
            "speech_duration": round(
                self.speech_duration,
                3,
            ),
            "speech_ratio": round(
                self.speech_ratio,
                4,
            ),
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "metadata": dict(
                self.metadata
            ),
        }



# ====================================================================
# FORMAT AUDIO PAR DEFAUT
# ====================================================================
#
# CORRECTIF : __init__.py et tests/test_diarization_global.py
# importaient DEFAULT_AUDIO_FORMAT depuis ce module, ou il n'a jamais
# ete defini. L'ImportError etait avalee par le try/except de
# __init__.py, ce qui mettait silencieusement a None TOUT le bloc des
# modeles : AudioCodec, AudioFormat, SessionStatus, SpeakerState,
# DiarizationEventType, SpeakerProfile, SpeakerSegment et
# DiarizationSession devenaient None alors qu'ils existent et
# fonctionnent.
#
# AudioFormat() sans argument decrit deja exactement le format interne
# SpeakerIdentity annonce en tete de fichier : PCM16, 16000 Hz, mono, 2 octets.
#
# ====================================================================

DEFAULT_AUDIO_FORMAT = AudioFormat()
