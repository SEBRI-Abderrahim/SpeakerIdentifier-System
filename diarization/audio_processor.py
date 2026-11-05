"""
====================================================================
SpeakerIdentity IA - Diarization Audio Processor
====================================================================

Projet       : SpeakerIdentity IA
Module       : Diarization
Fichier      : diarization/audio_processor.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 2.0.1
Date         : 14/08/2026

Description
------------

Processeur audio central du module de diarisation SpeakerIdentity IA.

Responsabilités :

    - validation des données audio ;
    - gestion du format PCM16 ;
    - conversion bytes <-> samples ;
    - calcul RMS ;
    - normalisation audio ;
    - découpage en frames ;
    - préparation du signal pour le VAD ;
    - préparation du signal pour les embeddings ;
    - calcul des durées audio ;
    - gestion des timestamps ;
    - statistiques de traitement.

Format interne SpeakerIdentity :

    Codec       : PCM16
    Sample rate : 16000 Hz
    Channels    : 1
    Width       : 2 bytes
    Endianness  : little-endian

Important
---------

Le processeur conserve uniquement le reliquat audio d'un flux
streaming afin de permettre la reconstruction correcte des frames VAD
entre plusieurs chunks WebSocket.

Il ne réalise PAS :

    - VAD ;
    - diarisation ;
    - identification speaker ;
    - génération d'embeddings ;
    - STT ;
    - resampling complexe.

====================================================================
"""

from __future__ import annotations


# ====================================================================
# IMPORTS
# ====================================================================

import math
import struct
import time

from dataclasses import dataclass
from typing import Any, Iterable, Optional


# ====================================================================
# INTERNAL IMPORTS
# ====================================================================

from .config import (
    SAMPLE_RATE,
    CHANNELS,
    SAMPLE_WIDTH,
    SAMPLE_FORMAT,
    VAD_FRAME_DURATION_MS,
)

from .exceptions import (
    AudioFormatError,
    AudioBufferError,
)

from .metrics import metrics


# ====================================================================
# CONSTANTS
# ====================================================================

PCM16_MIN = -32768
PCM16_MAX = 32767

BYTES_PER_PCM16_SAMPLE = 2

DEFAULT_SAMPLE_RATE = SAMPLE_RATE
DEFAULT_CHANNELS = CHANNELS
DEFAULT_SAMPLE_WIDTH = SAMPLE_WIDTH
DEFAULT_FRAME_DURATION_MS = VAD_FRAME_DURATION_MS


# ====================================================================
# DATA MODEL
# ====================================================================

@dataclass(frozen=True)
class AudioChunkInfo:
    """
    Informations descriptives concernant un bloc audio.
    """

    byte_size: int
    sample_count: int
    duration_seconds: float
    sample_rate: int
    channels: int
    sample_width: int
    rms: float
    peak: float
    timestamp: float

    @property
    def duration_ms(self) -> float:
        """
        Durée du bloc en millisecondes.
        """

        return self.duration_seconds * 1000.0

    def to_dict(self) -> dict:
        """
        Sérialise les informations.
        """

        return {
            "byte_size": self.byte_size,
            "sample_count": self.sample_count,
            "duration_seconds": round(
                self.duration_seconds,
                6,
            ),
            "duration_ms": round(
                self.duration_ms,
                3,
            ),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
            "rms": round(
                self.rms,
                6,
            ),
            "peak": round(
                self.peak,
                6,
            ),
            "timestamp": self.timestamp,
        }


@dataclass
class AudioProcessorStatistics:
    """
    Statistiques du processeur audio.
    """

    chunks_processed: int = 0
    bytes_processed: int = 0
    samples_processed: int = 0
    frames_created: int = 0
    invalid_chunks: int = 0
    clipped_samples: int = 0
    processing_time: float = 0.0
    started_at: float = 0.0

    def __post_init__(self) -> None:

        if self.started_at <= 0.0:
            self.started_at = time.time()

    @property
    def uptime(self) -> float:
        """
        Temps d'activité du processeur.
        """

        return max(
            0.0,
            time.time() - self.started_at,
        )

    def to_dict(self) -> dict:
        """
        Sérialise les statistiques.
        """

        return {
            "chunks_processed": self.chunks_processed,
            "bytes_processed": self.bytes_processed,
            "samples_processed": self.samples_processed,
            "frames_created": self.frames_created,
            "invalid_chunks": self.invalid_chunks,
            "clipped_samples": self.clipped_samples,
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
# AUDIO PROCESSOR
# ====================================================================

class AudioProcessor:
    """
    Processeur audio PCM16 pour SpeakerIdentity.

    Le processeur ne contient aucune logique de diarisation.

    Il assure uniquement :

        PCM16
          |
          +--> validation
          +--> conversion
          +--> frames
          +--> RMS
          +--> peak
          +--> statistiques

    Pour le streaming, le reliquat d'une frame est conservé afin
    qu'un chunk WebSocket incomplet puisse être complété par le
    chunk suivant.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        sample_width: int = DEFAULT_SAMPLE_WIDTH,
        frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS,
        config: Optional[Any] = None,
    ) -> None:

        # -------------------------------------------------------------
        # CONFIGURATION
        # -------------------------------------------------------------

        if config is not None:

            sample_rate = getattr(
                config,
                "sample_rate",
                sample_rate,
            )

            channels = getattr(
                config,
                "channels",
                channels,
            )

            sample_width = getattr(
                config,
                "sample_width",
                sample_width,
            )

            frame_duration_ms = getattr(
                config,
                "frame_duration_ms",
                getattr(
                    config,
                    "vad_frame_duration_ms",
                    frame_duration_ms,
                ),
            )

        self.sample_rate = int(
            sample_rate
        )

        self.channels = int(
            channels
        )

        self.sample_width = int(
            sample_width
        )

        self.frame_duration_ms = int(
            frame_duration_ms
        )

        self._validate_configuration()

        self.statistics = (
            AudioProcessorStatistics()
        )

        # -------------------------------------------------------------
        # STREAMING BUFFER
        # -------------------------------------------------------------

        self._frame_buffer = bytearray()

    # =================================================================
    # VALIDATION CONFIGURATION
    # =================================================================

    def _validate_configuration(self) -> None:
        """
        Valide la configuration du processeur.
        """

        if self.sample_rate <= 0:
            raise AudioFormatError(
                "sample_rate must be greater than zero."
            )

        if self.channels != 1:
            raise AudioFormatError(
                "SpeakerIdentity diarization requires mono audio."
            )

        if self.sample_width != 2:
            raise AudioFormatError(
                "SpeakerIdentity diarization requires PCM16 audio."
            )

        if self.frame_duration_ms not in (
            10,
            20,
            30,
        ):
            raise AudioFormatError(
                "frame_duration_ms must be 10, 20 or 30 ms."
            )

    # =================================================================
    # PROPERTIES
    # =================================================================

    @property
    def bytes_per_sample(self) -> int:
        """
        Nombre d'octets par échantillon.
        """

        return (
            self.sample_width
            * self.channels
        )

    @property
    def samples_per_frame(self) -> int:
        """
        Nombre d'échantillons dans une frame.
        """

        return int(
            self.sample_rate
            * self.frame_duration_ms
            / 1000
        )

    @property
    def frame_bytes(self) -> int:
        """
        Taille exacte d'une frame VAD.
        """

        return (
            self.samples_per_frame
            * self.bytes_per_sample
        )

    @property
    def frame_duration_seconds(self) -> float:
        """
        Durée d'une frame.
        """

        return (
            self.frame_duration_ms
            / 1000.0
        )

    @property
    def buffered_bytes(self) -> int:
        """
        Nombre d'octets actuellement conservés dans le buffer
        streaming.
        """

        return len(
            self._frame_buffer
        )

    # =================================================================
    # VALIDATE AUDIO
    # =================================================================

    def validate_audio(
        self,
        data: bytes | bytearray | memoryview,
        *,
        allow_empty: bool = False,
        require_complete_sample: bool = True,
    ) -> bytes:
        """
        Valide un bloc PCM16.
        """

        if data is None:
            raise AudioBufferError(
                "Audio data cannot be None."
            )

        if not isinstance(
            data,
            (
                bytes,
                bytearray,
                memoryview,
            ),
        ):
            raise AudioBufferError(
                "Audio data must be bytes, "
                "bytearray or memoryview."
            )

        raw = bytes(data)

        if not raw and not allow_empty:
            raise AudioBufferError(
                "Audio data cannot be empty."
            )

        if (
            require_complete_sample
            and len(raw) % self.bytes_per_sample != 0
        ):
            raise AudioFormatError(
                "PCM16 data contains an incomplete sample."
            )

        return raw

    # =================================================================
    # BYTES -> SAMPLES
    # =================================================================

    def bytes_to_samples(
        self,
        data: bytes | bytearray | memoryview,
    ) -> list[int]:
        """
        Convertit du PCM16 little-endian en échantillons Python.
        """

        raw = self.validate_audio(data)

        if not raw:
            return []

        sample_count = (
            len(raw)
            // self.bytes_per_sample
        )

        try:

            samples = struct.unpack(
                f"<{sample_count}h",
                raw,
            )

        except struct.error as exc:

            self.statistics.invalid_chunks += 1

            raise AudioFormatError(
                "Unable to decode PCM16 audio."
            ) from exc

        return list(samples)

    # =================================================================
    # SAMPLES -> BYTES
    # =================================================================

    def samples_to_bytes(
        self,
        samples: Iterable[int | float],
    ) -> bytes:
        """
        Convertit une séquence d'échantillons en PCM16.
        """

        if samples is None:
            raise AudioBufferError(
                "Samples cannot be None."
            )

        values = []

        for sample in samples:

            try:

                value = int(
                    round(
                        float(sample)
                    )
                )

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise AudioFormatError(
                    "Invalid audio sample value."
                ) from exc

            if value < PCM16_MIN:

                self.statistics.clipped_samples += 1

                value = PCM16_MIN

            elif value > PCM16_MAX:

                self.statistics.clipped_samples += 1

                value = PCM16_MAX

            values.append(value)

        if not values:
            return b""

        try:

            return struct.pack(
                f"<{len(values)}h",
                *values,
            )

        except struct.error as exc:

            raise AudioFormatError(
                "Unable to encode PCM16 audio."
            ) from exc

    # =================================================================
    # NORMALIZED SAMPLES
    # =================================================================

    def bytes_to_normalized_samples(
        self,
        data: bytes | bytearray | memoryview,
    ) -> list[float]:
        """
        Convertit PCM16 vers des valeurs normalisées.
        """

        samples = self.bytes_to_samples(
            data
        )

        if not samples:
            return []

        return [
            sample / 32768.0
            for sample in samples
        ]

    # =================================================================
    # NORMALIZED -> PCM16
    # =================================================================

    def normalized_samples_to_bytes(
        self,
        samples: Iterable[int | float],
    ) -> bytes:
        """
        Convertit des valeurs normalisées vers PCM16.
        """

        if samples is None:
            raise AudioBufferError(
                "Samples cannot be None."
            )

        pcm_samples = []

        for sample in samples:

            try:

                value = float(sample)

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise AudioFormatError(
                    "Invalid normalized sample."
                ) from exc

            value = max(
                -1.0,
                min(
                    1.0,
                    value,
                ),
            )

            if value >= 1.0:

                pcm = PCM16_MAX

            else:

                pcm = int(
                    round(
                        value * 32768.0
                    )
                )

            pcm_samples.append(
                pcm
            )

        return self.samples_to_bytes(
            pcm_samples
        )

    # =================================================================
    # RMS
    # =================================================================

    def calculate_rms(
        self,
        data: bytes | bytearray | memoryview,
    ) -> float:
        """
        Calcule le RMS normalisé d'un signal PCM16.
        """

        raw = self.validate_audio(
            data
        )

        if not raw:
            return 0.0

        samples = self.bytes_to_samples(
            raw
        )

        if not samples:
            return 0.0

        mean_square = sum(
            sample * sample
            for sample in samples
        ) / len(samples)

        rms = math.sqrt(
            mean_square
        )

        return rms / 32768.0

    # =================================================================
    # PEAK
    # =================================================================

    def calculate_peak(
        self,
        data: bytes | bytearray | memoryview,
    ) -> float:
        """
        Retourne le niveau peak normalisé.
        """

        raw = self.validate_audio(
            data
        )

        if not raw:
            return 0.0

        samples = self.bytes_to_samples(
            raw
        )

        if not samples:
            return 0.0

        peak = max(
            abs(sample)
            for sample in samples
        )

        return peak / 32768.0

    # =================================================================
    # AUDIO INFO
    # =================================================================

    def get_chunk_info(
        self,
        data: bytes | bytearray | memoryview,
    ) -> AudioChunkInfo:
        """
        Retourne les informations d'un bloc audio.
        """

        raw = self.validate_audio(
            data
        )

        sample_count = (
            len(raw)
            // self.bytes_per_sample
        )

        duration = (
            sample_count
            / self.sample_rate
        )

        return AudioChunkInfo(
            byte_size=len(raw),
            sample_count=sample_count,
            duration_seconds=duration,
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width=self.sample_width,
            rms=self.calculate_rms(raw),
            peak=self.calculate_peak(raw),
            timestamp=time.time(),
        )

    # =================================================================
    # DURATION
    # =================================================================

    def duration_seconds(
        self,
        data: bytes | bytearray | memoryview,
    ) -> float:
        """
        Calcule la durée d'un bloc audio.
        """

        raw = self.validate_audio(
            data
        )

        sample_count = (
            len(raw)
            // self.bytes_per_sample
        )

        return (
            sample_count
            / self.sample_rate
        )

    # =================================================================
    # SPLIT FRAMES - NON STREAMING
    # =================================================================

    def split_frames(
        self,
        data: bytes | bytearray | memoryview,
        *,
        include_partial: bool = False,
    ) -> list[bytes]:
        """
        Découpe un bloc audio en frames VAD.

        Cette méthode ne conserve PAS le reliquat.

        Pour le streaming temps réel, utiliser iter_frames().
        """

        raw = self.validate_audio(
            data
        )

        if not raw:
            return []

        frames = []

        offset = 0

        while (
            offset + self.frame_bytes
            <= len(raw)
        ):

            frames.append(
                raw[
                    offset:
                    offset + self.frame_bytes
                ]
            )

            offset += self.frame_bytes

        if (
            include_partial
            and offset < len(raw)
        ):

            partial = raw[
                offset:
            ]

            frames.append(
                partial
            )

        self.statistics.frames_created += (
            len(frames)
        )

        return frames

    # =================================================================
    # ITER FRAMES - STREAMING
    # =================================================================

    def iter_frames(
        self,
        data: bytes | bytearray | memoryview,
        *,
        include_partial: bool = False,
    ):
        """
        Produit les frames VAD à partir d'un flux streaming.

        Correction importante :

        Les chunks WebSocket ne correspondent pas forcément
        exactement à une frame VAD.

        Exemple :

            chunk 1 = 320 bytes
            chunk 2 = 640 bytes
            chunk 3 = 300 bytes

        Le processeur conserve automatiquement les octets
        incomplets entre les appels.

        Les frames produites sont toujours de taille exacte :

            10 ms  -> 320 bytes
            20 ms  -> 640 bytes
            30 ms  -> 960 bytes

        Le reliquat reste dans _frame_buffer.
        """

        raw = self.validate_audio(
            data
        )

        if raw:
            self._frame_buffer.extend(
                raw
            )

        while (
            len(self._frame_buffer)
            >= self.frame_bytes
        ):

            frame = bytes(
                self._frame_buffer[
                    :self.frame_bytes
                ]
            )

            del self._frame_buffer[
                :self.frame_bytes
            ]

            self.statistics.frames_created += 1

            yield frame

        # -------------------------------------------------------------
        # PARTIAL
        # -------------------------------------------------------------

        if (
            include_partial
            and self._frame_buffer
        ):

            partial = bytes(
                self._frame_buffer
            )

            self._frame_buffer.clear()

            yield partial

    # =================================================================
    # FLUSH FRAME BUFFER
    # =================================================================

    def flush_frames(
        self,
    ) -> Optional[bytes]:
        """
        Retourne le reliquat du buffer streaming.

        Le reliquat est retourné tel quel.

        Attention :

        Le VAD ne doit généralement pas recevoir une frame
        incomplète. Cette méthode sert uniquement à la fin
        d'une session pour permettre au niveau supérieur de
        décider quoi faire du reliquat.
        """

        if not self._frame_buffer:
            return None

        remainder = bytes(
            self._frame_buffer
        )

        self._frame_buffer.clear()

        return remainder

    # =================================================================
    # CLEAR FRAME BUFFER
    # =================================================================

    def clear_frame_buffer(self) -> None:
        """
        Supprime le reliquat audio streaming.
        """

        self._frame_buffer.clear()

    # =================================================================
    # MERGE FRAMES
    # =================================================================

    def merge_frames(
        self,
        frames: Iterable[
            bytes | bytearray | memoryview
        ],
    ) -> bytes:
        """
        Fusionne plusieurs frames PCM16.
        """

        if frames is None:
            raise AudioBufferError(
                "frames cannot be None."
            )

        output = bytearray()

        for frame in frames:

            raw = self.validate_audio(
                frame
            )

            output.extend(
                raw
            )

        return bytes(output)

    # =================================================================
    # NORMALIZE
    # =================================================================

    def normalize(
        self,
        data: bytes | bytearray | memoryview,
        target_peak: float = 0.95,
    ) -> bytes:
        """
        Normalise le niveau peak du signal.
        """

        if not (
            0.0 < target_peak <= 1.0
        ):
            raise ValueError(
                "target_peak must be between "
                "0 and 1."
            )

        raw = self.validate_audio(
            data
        )

        if not raw:
            return raw

        samples = self.bytes_to_samples(
            raw
        )

        if not samples:
            return b""

        peak = max(
            abs(sample)
            for sample in samples
        )

        if peak == 0:
            return raw

        current_peak = (
            peak / 32768.0
        )

        gain = (
            target_peak
            / current_peak
        )

        normalized = [
            sample * gain
            for sample in samples
        ]

        return self.samples_to_bytes(
            normalized
        )

    # =================================================================
    # REMOVE DC OFFSET
    # =================================================================

    def remove_dc_offset(
        self,
        data: bytes | bytearray | memoryview,
    ) -> bytes:
        """
        Supprime approximativement la composante DC.
        """

        raw = self.validate_audio(
            data
        )

        if not raw:
            return raw

        samples = self.bytes_to_samples(
            raw
        )

        if not samples:
            return b""

        mean = (
            sum(samples)
            / len(samples)
        )

        corrected = [
            sample - mean
            for sample in samples
        ]

        return self.samples_to_bytes(
            corrected
        )

    # =================================================================
    # SILENCE
    # =================================================================

    def is_silence(
        self,
        data: bytes | bytearray | memoryview,
        threshold: float = 0.008,
    ) -> bool:
        """
        Détermine si le bloc peut être considéré
        comme silencieux selon son RMS.

        Ce test n'est PAS le VAD principal.
        """

        if threshold < 0:
            raise ValueError(
                "threshold must be >= 0."
            )

        return (
            self.calculate_rms(data)
            < threshold
        )

    # =================================================================
    # PROCESS CHUNK
    # =================================================================

    def process(
        self,
        data: bytes | bytearray | memoryview,
        *,
        normalize: bool = False,
        remove_dc: bool = False,
    ) -> bytes:
        """
        Valide et prépare un bloc PCM16.

        Le résultat est directement compatible avec
        DiarizationPipeline.
        """

        started = time.perf_counter()

        try:

            raw = self.validate_audio(
                data
            )

            if remove_dc:

                raw = self.remove_dc_offset(
                    raw
                )

            if normalize:

                raw = self.normalize(
                    raw
                )

            info = self.get_chunk_info(
                raw
            )

            self.statistics.chunks_processed += 1

            self.statistics.bytes_processed += (
                len(raw)
            )

            self.statistics.samples_processed += (
                info.sample_count
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            self.statistics.processing_time += (
                elapsed
            )

            # ---------------------------------------------------------
            # GLOBAL METRICS
            # ---------------------------------------------------------

            metrics.increment(
                "audio_chunks",
                1,
            )

            metrics.increment(
                "audio_bytes",
                len(raw),
            )

            metrics.add_processing_time(
                elapsed
            )

            return raw

        except (
            AudioFormatError,
            AudioBufferError,
        ):

            self.statistics.invalid_chunks += 1

            raise

    # =================================================================
    # RESET STATISTICS
    # =================================================================

    def reset_statistics(self) -> None:
        """
        Réinitialise les statistiques locales.
        """

        self.statistics = (
            AudioProcessorStatistics()
        )

        self.clear_frame_buffer()

    # =================================================================
    # HEALTH
    # =================================================================

    def health(self) -> dict:
        """
        Health check du processeur.
        """

        return {
            "status": "healthy",
            "component": "audio_processor",
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
            "format": SAMPLE_FORMAT,
            "codec": SAMPLE_FORMAT,
            "frame_duration_ms": (
                self.frame_duration_ms
            ),
            "frame_bytes": self.frame_bytes,
            "buffered_bytes": self.buffered_bytes,
            "statistics": (
                self.statistics.to_dict()
            ),
        }


# ====================================================================
# DEFAULT PROCESSOR FACTORY
# ====================================================================

def create_audio_processor() -> AudioProcessor:
    """
    Crée un AudioProcessor utilisant la configuration SpeakerIdentity.
    """

    return AudioProcessor(
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        sample_width=SAMPLE_WIDTH,
        frame_duration_ms=VAD_FRAME_DURATION_MS,
    )


# ====================================================================
# DEFAULT INSTANCE
# ====================================================================

audio_processor = create_audio_processor()


# ====================================================================
# PUBLIC API
# ====================================================================

__all__ = [
    "PCM16_MIN",
    "PCM16_MAX",
    "BYTES_PER_PCM16_SAMPLE",
    "AudioChunkInfo",
    "AudioProcessorStatistics",
    "AudioProcessor",
    "create_audio_processor",
    "audio_processor",
]