"""
====================================================================
SpeakerIdentity IA - Diarization Voice Activity Detection
====================================================================

Projet       : SpeakerIdentity IA
Module       : Diarization
Fichier      : diarization/vad.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 2.1.1
Date         : 14/08/2026

Description
------------

Détection d'activité vocale du module de diarisation SpeakerIdentity IA.

Responsabilités :

    - VAD WebRTC ;
    - fallback RMS ;
    - détection speech / silence ;
    - détection début de parole ;
    - détection fin de parole ;
    - traitement temps réel ;
    - gestion des frames PCM16 ;
    - statistiques VAD.

Format audio interne :

    Codec       : PCM16
    Sample rate : 16000 Hz
    Channels    : 1
    Width       : 2 bytes

Compatibilité :

    Le constructeur accepte également un paramètre ``config``
    afin de rester compatible avec DiarizationPipeline.

====================================================================
"""

from __future__ import annotations

import logging
import time

from dataclasses import dataclass
from typing import Any, Optional


from .config import (
    SAMPLE_RATE,
    CHANNELS,
    SAMPLE_WIDTH,
    VAD_FRAME_DURATION_MS,
    VAD_ENABLED,
    VAD_BACKEND,
    VAD_AGGRESSIVENESS,
    SILENCE_RMS,
    MIN_SPEECH_MS,
    MIN_SILENCE_MS,
    PADDING_MS,
)

from .exceptions import AudioFormatError

from .metrics import metrics


# ====================================================================
# LOGGER
# ====================================================================

logger = logging.getLogger(
    "speakeridentity.diarization.vad"
)


# Repli RMS deja signale ? (un seul avertissement par processus)
_REPLI_RMS_SIGNALE = False


# ====================================================================
# CONSTANTS
# ====================================================================

PCM16_BYTES_PER_SAMPLE = 2

SUPPORTED_FRAME_DURATIONS = (
    10,
    20,
    30,
)

SUPPORTED_SAMPLE_RATE = 16000


# ====================================================================
# OPTIONAL WEBRTC IMPORT
# ====================================================================

try:

    import webrtcvad

    WEBRTC_AVAILABLE = True

except ImportError:

    webrtcvad = None

    WEBRTC_AVAILABLE = False


# ====================================================================
# VAD RESULT
# ====================================================================

@dataclass(frozen=True)
class VADResult:

    is_speech: bool

    timestamp: float

    duration_seconds: float

    rms: float

    backend: str

    confidence: float = 0.0

    frame_size: int = 0

    reason: str = ""

    def to_dict(self) -> dict:

        return {
            "is_speech": self.is_speech,
            "timestamp": self.timestamp,
            "duration_seconds": round(
                self.duration_seconds,
                6,
            ),
            "rms": round(
                self.rms,
                6,
            ),
            "backend": self.backend,
            "confidence": round(
                max(
                    0.0,
                    min(
                        1.0,
                        self.confidence,
                    ),
                ),
                4,
            ),
            "frame_size": self.frame_size,
            "reason": self.reason,
        }


# ====================================================================
# VAD SEGMENT
# ====================================================================

@dataclass
class VADSegment:

    start: float

    end: Optional[float] = None

    confidence: float = 0.0

    frame_count: int = 0

    closed: bool = False

    def close(
        self,
        end: float,
    ) -> None:

        self.end = max(
            self.start,
            float(end),
        )

        self.closed = True

    @property
    def duration(self) -> float:

        if self.end is None:

            return 0.0

        return max(
            0.0,
            self.end - self.start,
        )

    def to_dict(self) -> dict:

        return {
            "start": self.start,
            "end": self.end,
            "duration": round(
                self.duration,
                6,
            ),
            "confidence": round(
                self.confidence,
                4,
            ),
            "frame_count": self.frame_count,
            "closed": self.closed,
        }


# ====================================================================
# VAD STATISTICS
# ====================================================================

@dataclass
class VADStatistics:

    frames_processed: int = 0

    speech_frames: int = 0

    silence_frames: int = 0

    speech_events: int = 0

    silence_events: int = 0

    segments_created: int = 0

    invalid_frames: int = 0

    webrtc_frames: int = 0

    rms_frames: int = 0

    processing_time: float = 0.0

    started_at: float = 0.0

    def __post_init__(self) -> None:

        if self.started_at <= 0:

            self.started_at = time.time()

    @property
    def uptime(self) -> float:

        return max(
            0.0,
            time.time() - self.started_at,
        )

    @property
    def speech_ratio(self) -> float:

        if self.frames_processed <= 0:

            return 0.0

        return (
            self.speech_frames
            / self.frames_processed
        )

    def to_dict(self) -> dict:

        return {
            "frames_processed": self.frames_processed,
            "speech_frames": self.speech_frames,
            "silence_frames": self.silence_frames,
            "speech_events": self.speech_events,
            "silence_events": self.silence_events,
            "segments_created": self.segments_created,
            "invalid_frames": self.invalid_frames,
            "webrtc_frames": self.webrtc_frames,
            "rms_frames": self.rms_frames,
            "speech_ratio": round(
                self.speech_ratio,
                4,
            ),
            "processing_time": round(
                self.processing_time,
                6,
            ),
            "uptime": round(
                self.uptime,
                3,
            ),
        }


# ====================================================================
# VOICE ACTIVITY DETECTOR
# ====================================================================

class VoiceActivityDetector:

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        sample_width: int = SAMPLE_WIDTH,
        frame_duration_ms: int = VAD_FRAME_DURATION_MS,
        backend: str = VAD_BACKEND,
        aggressiveness: int = VAD_AGGRESSIVENESS,
        rms_threshold: float = SILENCE_RMS,
        min_speech_ms: int = MIN_SPEECH_MS,
        min_silence_ms: int = MIN_SILENCE_MS,
        padding_ms: int = PADDING_MS,
        enabled: bool = VAD_ENABLED,
        config: Optional[Any] = None,
    ) -> None:

        # -------------------------------------------------------------
        # Configuration object compatibility
        # -------------------------------------------------------------

        self.config = config

        if config is not None:

            sample_rate = self._config_value(
                config,
                "SAMPLE_RATE",
                "sample_rate",
                default=sample_rate,
            )

            channels = self._config_value(
                config,
                "CHANNELS",
                "channels",
                default=channels,
            )

            sample_width = self._config_value(
                config,
                "SAMPLE_WIDTH",
                "sample_width",
                default=sample_width,
            )

            frame_duration_ms = self._config_value(
                config,
                "VAD_FRAME_DURATION_MS",
                "vad_frame_duration_ms",
                default=frame_duration_ms,
            )

            backend = self._config_value(
                config,
                "VAD_BACKEND",
                "vad_backend",
                default=backend,
            )

            aggressiveness = self._config_value(
                config,
                "VAD_AGGRESSIVENESS",
                "vad_aggressiveness",
                default=aggressiveness,
            )

            rms_threshold = self._config_value(
                config,
                "SILENCE_RMS",
                "silence_rms",
                default=rms_threshold,
            )

            min_speech_ms = self._config_value(
                config,
                "MIN_SPEECH_MS",
                "min_speech_ms",
                default=min_speech_ms,
            )

            min_silence_ms = self._config_value(
                config,
                "MIN_SILENCE_MS",
                "min_silence_ms",
                default=min_silence_ms,
            )

            padding_ms = self._config_value(
                config,
                "PADDING_MS",
                "padding_ms",
                default=padding_ms,
            )

            enabled = self._config_value(
                config,
                "VAD_ENABLED",
                "vad_enabled",
                default=enabled,
            )

        self.sample_rate = int(sample_rate)

        self.channels = int(channels)

        self.sample_width = int(sample_width)

        self.frame_duration_ms = int(
            frame_duration_ms
        )

        self.backend = str(
            backend
        ).strip().lower()

        self.aggressiveness = int(
            aggressiveness
        )

        self.rms_threshold = float(
            rms_threshold
        )

        self.min_speech_ms = int(
            min_speech_ms
        )

        self.min_silence_ms = int(
            min_silence_ms
        )

        self.padding_ms = int(
            padding_ms
        )

        self.enabled = bool(enabled)

        self.statistics = VADStatistics()

        self._current_segment: Optional[VADSegment] = None

        self._speech_duration = 0.0

        self._silence_duration = 0.0

        self._last_timestamp: Optional[float] = None

        self._webrtc = None

        self._validate_configuration()

        self._initialize_backend()

    # =================================================================
    # CONFIGURATION HELPER
    # =================================================================

    @staticmethod
    def _config_value(
        config: Any,
        *names: str,
        default: Any = None,
    ) -> Any:
        """
        Récupère une valeur depuis un objet de configuration.

        Compatible avec :

            - objet Python ;
            - classe de configuration ;
            - dictionnaire.

        Les paramètres explicitement fournis au constructeur
        restent utilisés lorsque la configuration ne contient
        pas la valeur recherchée.
        """

        if config is None:

            return default

        for name in names:

            if isinstance(config, dict):

                if name in config:

                    value = config[name]

                    if value is not None:

                        return value

            else:

                if hasattr(config, name):

                    value = getattr(
                        config,
                        name,
                    )

                    if value is not None:

                        return value

        return default

    # =================================================================
    # VALIDATION
    # =================================================================

    def _validate_configuration(self) -> None:

        if self.sample_rate != SUPPORTED_SAMPLE_RATE:

            raise AudioFormatError(
                "SpeakerIdentity WebRTC VAD requires "
                "16000 Hz audio."
            )

        if self.channels != 1:

            raise AudioFormatError(
                "VAD requires mono audio."
            )

        if self.sample_width != PCM16_BYTES_PER_SAMPLE:

            raise AudioFormatError(
                "VAD requires PCM16 audio."
            )

        if self.frame_duration_ms not in SUPPORTED_FRAME_DURATIONS:

            raise AudioFormatError(
                "VAD frame duration must be "
                "10, 20 or 30 ms."
            )

        if not 0 <= self.aggressiveness <= 3:

            raise ValueError(
                "aggressiveness must be "
                "between 0 and 3."
            )

        if self.rms_threshold < 0:

            raise ValueError(
                "rms_threshold must be >= 0."
            )

        if self.min_speech_ms < 0:

            raise ValueError(
                "min_speech_ms must be >= 0."
            )

        if self.min_silence_ms < 0:

            raise ValueError(
                "min_silence_ms must be >= 0."
            )

        if self.padding_ms < 0:

            raise ValueError(
                "padding_ms must be >= 0."
            )

    # =================================================================
    # BACKEND
    # =================================================================

    def _initialize_backend(self) -> None:

        requested = self.backend

        if requested not in {
            "webrtc",
            "rms",
            "auto",
        }:

            raise ValueError(
                "Unsupported VAD backend: "
                f"{requested}"
            )

        if not self.enabled:

            self.backend = "disabled"

            self._webrtc = None

            return

        if (
            requested in {
                "webrtc",
                "auto",
            }
            and WEBRTC_AVAILABLE
        ):

            try:

                self._webrtc = webrtcvad.Vad(
                    self.aggressiveness
                )

                self.backend = "webrtc"

                return

            except Exception as exc:

                logger.warning(
                    "Unable to initialize WebRTC VAD: %s",
                    exc,
                )

        # CORRECTIF : l'avertissement n'etait emis que si le backend
        # "webrtc" avait ete demande explicitement. Or VAD_BACKEND vaut
        # "auto" par defaut : sur un poste sans webrtcvad, le module
        # basculait donc EN SILENCE sur un simple seuil d'energie.
        #
        # Ce repli n'est pas un detecteur de voix : il classe le bruit
        # de fond comme parole et un chuchotement comme silence. Les
        # segments envoyes au moteur d'embeddings contiennent alors du
        # bruit, ce qui rend l'identification incoherente. Pire, avec un
        # bruit de fond constant, is_speech reste vrai en permanence.
        #
        # L'avertissement est desormais emis dans tous les cas, une
        # seule fois par processus pour ne pas noyer les journaux.

        global _REPLI_RMS_SIGNALE

        if not _REPLI_RMS_SIGNALE:

            _REPLI_RMS_SIGNALE = True

            logger.warning(
                "VAD : webrtcvad indisponible, repli sur un simple "
                "seuil d'energie RMS (backend demande : %s). "
                "Ce repli ne distingue pas la voix du bruit et degrade "
                "fortement l'identification d'interlocuteur. "
                "Installer le vrai VAD :  pip install webrtcvad",
                requested,
            )

        self.backend = "rms"

        self._webrtc = None

    # =================================================================
    # PROPERTIES
    # =================================================================

    @property
    def samples_per_frame(self) -> int:

        return int(
            self.sample_rate
            * self.frame_duration_ms
            / 1000
        )

    @property
    def frame_bytes(self) -> int:

        return (
            self.samples_per_frame
            * self.sample_width
            * self.channels
        )

    @property
    def frame_duration_seconds(self) -> float:

        return (
            self.frame_duration_ms
            / 1000.0
        )

    @property
    def is_speech_active(self) -> bool:

        return (
            self._current_segment is not None
            and not self._current_segment.closed
        )

    # =================================================================
    # FRAME VALIDATION
    # =================================================================

    def _validate_frame(
        self,
        frame,
    ) -> bytes:

        if frame is None:

            raise AudioFormatError(
                "VAD frame cannot be None."
            )

        if not isinstance(
            frame,
            (
                bytes,
                bytearray,
                memoryview,
            ),
        ):

            raise AudioFormatError(
                "VAD frame must be bytes-like."
            )

        raw = bytes(frame)

        if not raw:

            raise AudioFormatError(
                "VAD frame cannot be empty."
            )

        if len(raw) != self.frame_bytes:

            raise AudioFormatError(
                "Invalid VAD frame size: "
                f"expected {self.frame_bytes} "
                f"bytes, got {len(raw)}."
            )

        return raw

    # =================================================================
    # TIMESTAMP VALIDATION
    # =================================================================

    @staticmethod
    def _validate_timestamp(
        timestamp: Optional[float],
        field_name: str,
    ) -> float:

        if timestamp is None:

            return time.time()

        try:

            value = float(timestamp)

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise AudioFormatError(
                f"{field_name} must be "
                "a valid number."
            ) from exc

        if value < 0:

            raise AudioFormatError(
                f"{field_name} must be "
                ">= 0."
            )

        return value

    # =================================================================
    # RMS
    # =================================================================

    @staticmethod
    def _calculate_rms(
        frame: bytes,
    ) -> float:

        if not frame:

            return 0.0

        sample_count = (
            len(frame)
            // PCM16_BYTES_PER_SAMPLE
        )

        if sample_count <= 0:

            return 0.0

        total = 0

        for index in range(
            0,
            len(frame),
            PCM16_BYTES_PER_SAMPLE,
        ):

            sample = int.from_bytes(
                frame[
                    index:
                    index + PCM16_BYTES_PER_SAMPLE
                ],
                byteorder="little",
                signed=True,
            )

            total += sample * sample

        mean_square = (
            total / sample_count
        )

        return (
            mean_square ** 0.5
        ) / 32768.0

    # =================================================================
    # WEBRTC
    # =================================================================

    def _detect_webrtc(
        self,
        frame: bytes,
    ) -> bool:

        if self._webrtc is None:

            raise RuntimeError(
                "WebRTC VAD is not initialized."
            )

        result = bool(
            self._webrtc.is_speech(
                frame,
                self.sample_rate,
            )
        )

        self.statistics.webrtc_frames += 1

        return result

    # =================================================================
    # RMS
    # =================================================================

    def _detect_rms(
        self,
        rms: float,
    ) -> bool:

        self.statistics.rms_frames += 1

        return rms >= self.rms_threshold

    # =================================================================
    # CONFIDENCE
    # =================================================================

    def _calculate_confidence(
        self,
        is_speech: bool,
        rms: float,
    ) -> float:

        if not is_speech:

            return 0.0

        if self.rms_threshold <= 0:

            return 1.0

        ratio = (
            rms
            / self.rms_threshold
        )

        return max(
            0.0,
            min(
                1.0,
                ratio / 4.0,
            ),
        )

    # =================================================================
    # DETECT
    # =================================================================

    def detect(
        self,
        frame,
        timestamp: Optional[float] = None,
    ) -> VADResult:

        started = time.perf_counter()

        try:

            raw = self._validate_frame(
                frame
            )

        except AudioFormatError:

            self.statistics.invalid_frames += 1

            raise

        now = self._validate_timestamp(
            timestamp,
            "VAD timestamp",
        )

        rms = self._calculate_rms(
            raw
        )

        if not self.enabled:

            is_speech = False

            backend = "disabled"

            reason = "vad_disabled"

        elif self.backend == "webrtc":

            try:

                is_speech = self._detect_webrtc(
                    raw
                )

                backend = "webrtc"

                reason = "webrtc"

            except Exception as exc:

                logger.warning(
                    "WebRTC VAD failed: %s. "
                    "Using RMS fallback.",
                    exc,
                )

                is_speech = self._detect_rms(
                    rms
                )

                backend = "rms_fallback"

                reason = (
                    "webrtc_error_rms_fallback"
                )

        else:

            is_speech = self._detect_rms(
                rms
            )

            backend = "rms"

            reason = "rms_threshold"

        confidence = self._calculate_confidence(
            is_speech,
            rms,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        self.statistics.processing_time += (
            elapsed
        )

        return VADResult(
            is_speech=is_speech,
            timestamp=now,
            duration_seconds=self.frame_duration_seconds,
            rms=rms,
            backend=backend,
            confidence=confidence,
            frame_size=len(raw),
            reason=reason,
        )

    # =================================================================
    # PROCESS FRAME
    # =================================================================

    def process_frame(
        self,
        frame,
        timestamp: Optional[float] = None,
    ) -> VADResult:

        result = self.detect(
            frame,
            timestamp,
        )

        self.statistics.frames_processed += 1

        if result.is_speech:

            self.statistics.speech_frames += 1

            self._speech_duration += (
                result.duration_seconds
            )

            self._silence_duration = 0.0

        else:

            self.statistics.silence_frames += 1

            self._silence_duration += (
                result.duration_seconds
            )

            self._speech_duration = 0.0

        metrics.increment(
            "frames_processed",
            1,
        )

        if result.is_speech:

            metrics.increment(
                "speech_frames",
                1,
            )

        else:

            metrics.increment(
                "silence_frames",
                1,
            )

        self._update_state(
            result
        )

        self._last_timestamp = (
            result.timestamp
        )

        return result

    # =================================================================
    # STATE
    # =================================================================

    def _update_state(
        self,
        result: VADResult,
    ) -> None:

        if result.is_speech:

            if (
                self._current_segment is not None
                and self._current_segment.closed
            ):

                return

            if self._current_segment is None:

                if (
                    self._speech_duration
                    >= self.min_speech_ms / 1000.0
                ):

                    start = (
                        result.timestamp
                        + result.duration_seconds
                        - self._speech_duration
                    )

                    start -= (
                        self.padding_ms / 1000.0
                    )

                    start = max(
                        0.0,
                        start,
                    )

                    accumulated_frames = max(
                        1,
                        int(
                            round(
                                self._speech_duration
                                / self.frame_duration_seconds
                            )
                        ),
                    )

                    self._current_segment = VADSegment(
                        start=start,
                        confidence=result.confidence,
                        frame_count=accumulated_frames,
                    )

                    self.statistics.speech_events += 1

            else:

                self._current_segment.frame_count += 1

                self._current_segment.confidence = max(
                    self._current_segment.confidence,
                    result.confidence,
                )

            return

        if self._current_segment is None:

            return

        if self._current_segment.closed:

            return

        if (
            self._silence_duration
            >= self.min_silence_ms / 1000.0
        ):

            end = (
                result.timestamp
                + result.duration_seconds
            )

            end += (
                self.padding_ms / 1000.0
            )

            self._current_segment.close(
                end
            )

            self.statistics.silence_events += 1

            self.statistics.segments_created += 1

            metrics.increment(
                "segments_created",
                1,
            )

    # =================================================================
    # CONSUME
    # =================================================================

    def consume_closed_segment(
        self,
    ) -> Optional[VADSegment]:

        segment = self._current_segment

        if (
            segment is not None
            and segment.closed
        ):

            self._current_segment = None

            self._speech_duration = 0.0

            self._silence_duration = 0.0

            return segment

        return None

    # =================================================================
    # FLUSH
    # =================================================================

    def flush(
        self,
        timestamp: Optional[float] = None,
    ) -> Optional[VADSegment]:

        if self._current_segment is None:

            return None

        if self._current_segment.closed:

            return self.consume_closed_segment()

        end = self._validate_timestamp(
            timestamp,
            "VAD timestamp",
        )

        if (
            self._last_timestamp is not None
            and end < self._last_timestamp
        ):

            end = self._last_timestamp

        end = max(
            self._current_segment.start,
            end,
        )

        self._current_segment.close(
            end
        )

        self.statistics.segments_created += 1

        metrics.increment(
            "segments_created",
            1,
        )

        return self.consume_closed_segment()

    # =================================================================
    # PROCESS CHUNK
    # =================================================================

    def process_chunk(
        self,
        audio,
        start_timestamp: Optional[float] = None,
    ) -> list[VADResult]:

        if audio is None:

            raise AudioFormatError(
                "Audio chunk cannot be None."
            )

        if not isinstance(
            audio,
            (
                bytes,
                bytearray,
                memoryview,
            ),
        ):

            raise AudioFormatError(
                "Audio chunk must be bytes-like."
            )

        raw = bytes(audio)

        if not raw:

            return []

        if len(raw) % self.frame_bytes != 0:

            raise AudioFormatError(
                "Audio chunk size must be a "
                "multiple of VAD frame size."
            )

        timestamp = self._validate_timestamp(
            start_timestamp,
            "VAD start_timestamp",
        )

        results: list[VADResult] = []

        for offset in range(
            0,
            len(raw),
            self.frame_bytes,
        ):

            frame = raw[
                offset:
                offset + self.frame_bytes
            ]

            result = self.process_frame(
                frame,
                timestamp,
            )

            results.append(
                result
            )

            timestamp += (
                self.frame_duration_seconds
            )

        return results

    # =================================================================
    # RESET
    # =================================================================

    def reset(self) -> None:

        self._current_segment = None

        self._speech_duration = 0.0

        self._silence_duration = 0.0

        self._last_timestamp = None

    # =================================================================
    # RESET STATISTICS
    # =================================================================

    def reset_statistics(self) -> None:

        self.statistics = VADStatistics()

    # =================================================================
    # HEALTH
    # =================================================================

    def health(self) -> dict:

        if not self.enabled:

            status = "disabled"

        elif self.backend == "webrtc":

            status = "healthy"

        elif self.backend == "rms":

            status = "degraded"

        else:

            status = "unhealthy"

        return {
            "status": status,
            "component": "vad",
            "enabled": self.enabled,
            "backend": self.backend,
            "webrtc_available": WEBRTC_AVAILABLE,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
            "frame_duration_ms": self.frame_duration_ms,
            "frame_bytes": self.frame_bytes,
            "aggressiveness": self.aggressiveness,
            "rms_threshold": self.rms_threshold,
            "speech_active": self.is_speech_active,
            "statistics": self.statistics.to_dict(),
        }

    # =================================================================
    # REPRESENTATION
    # =================================================================

    def __repr__(self) -> str:

        return (
            "<VoiceActivityDetector "
            f"backend={self.backend!r}, "
            f"sample_rate={self.sample_rate}, "
            f"frame_ms={self.frame_duration_ms}, "
            f"speech_active="
            f"{self.is_speech_active}>"
        )


# ====================================================================
# FACTORY
# ====================================================================

def create_vad(
    **overrides,
) -> VoiceActivityDetector:

    return VoiceActivityDetector(
        sample_rate=overrides.pop(
            "sample_rate",
            SAMPLE_RATE,
        ),
        channels=overrides.pop(
            "channels",
            CHANNELS,
        ),
        sample_width=overrides.pop(
            "sample_width",
            SAMPLE_WIDTH,
        ),
        frame_duration_ms=overrides.pop(
            "frame_duration_ms",
            VAD_FRAME_DURATION_MS,
        ),
        backend=overrides.pop(
            "backend",
            VAD_BACKEND,
        ),
        aggressiveness=overrides.pop(
            "aggressiveness",
            VAD_AGGRESSIVENESS,
        ),
        rms_threshold=overrides.pop(
            "rms_threshold",
            SILENCE_RMS,
        ),
        min_speech_ms=overrides.pop(
            "min_speech_ms",
            MIN_SPEECH_MS,
        ),
        min_silence_ms=overrides.pop(
            "min_silence_ms",
            MIN_SILENCE_MS,
        ),
        padding_ms=overrides.pop(
            "padding_ms",
            PADDING_MS,
        ),
        enabled=overrides.pop(
            "enabled",
            VAD_ENABLED,
        ),
        config=overrides.pop(
            "config",
            None,
        ),
        **overrides,
    )


# ====================================================================
# DEFAULT INSTANCE
# ====================================================================

vad = create_vad()

voice_activity_detector = vad


# ====================================================================
# PUBLIC API
# ====================================================================

__all__ = [
    "WEBRTC_AVAILABLE",
    "SUPPORTED_FRAME_DURATIONS",
    "VADResult",
    "VADSegment",
    "VADStatistics",
    "VoiceActivityDetector",
    "create_vad",
    "vad",
    "voice_activity_detector",
]