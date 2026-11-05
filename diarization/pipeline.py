"""
====================================================================
SpeakerIdentity IA - Diarization Pipeline
====================================================================

Projet       : SpeakerIdentity IA
Module       : Diarization
Fichier      : diarization/pipeline.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 2.0.7
Date         : 15/08/2026

Description
------------

Pipeline principal de diarisation SpeakerIdentity IA.

Responsabilités :

    - gestion des sessions ;
    - traitement audio ;
    - VAD ;
    - segmentation ;
    - extraction des embeddings ;
    - identification des speakers ;
    - génération des événements ;
    - statistiques ;
    - health check.

Format audio interne :

    PCM16
    16000 Hz
    Mono

====================================================================
"""

from __future__ import annotations

import logging
import time

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


from .config import (
    DiarizationConfig,
    config as default_config,
)

from .exceptions import (
    AudioFormatError,
    DiarizationError,
    EmbeddingError,
    PipelineError,
    SessionError,
    SessionNotFoundError,
)

from .events import (
    DiarizationEvent,
    EventTypes,
)

from .metrics import (
    DiarizationMetrics,
    metrics as default_metrics,
)

from .audio_processor import (
    AudioProcessor,
)

from .vad import (
    VoiceActivityDetector,
)

from .segment_manager import (
    SegmentManager,
)

from .embedding_engine import (
    EmbeddingEngine,
)

from .speaker_manager import (
    SpeakerManager,
)

from .session_manager import (
    DiarizationSessionManager,
)


# ====================================================================
# LOGGER
# ====================================================================

logger = logging.getLogger(
    "speakeridentity.diarization.pipeline"
)


# ====================================================================
# CONSTANTES
# ====================================================================

EMBEDDING_MAX_DURATION_MS = 3000


# ====================================================================
# PIPELINE RESULT
# ====================================================================

@dataclass
class PipelineResult:
    """
    Résultat d'un traitement du pipeline.
    """

    session_id: str

    segments: List[Any]

    events: List[DiarizationEvent]

    audio_duration: float = 0.0

    speech_duration: float = 0.0

    frames_processed: int = 0

    speech_frames: int = 0

    silence_frames: int = 0

    processing_time: float = 0.0

    created_at: float = field(
        default_factory=time.time
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        audio_duration = max(
            0.0,
            self.audio_duration,
        )

        speech_duration = min(
            max(
                0.0,
                self.speech_duration,
            ),
            audio_duration,
        )

        return {
            "session_id": self.session_id,

            "segments": [
                (
                    segment.to_dict()
                    if hasattr(segment, "to_dict")
                    else segment
                )
                for segment in self.segments
            ],

            "events": [
                (
                    event.to_dict()
                    if hasattr(event, "to_dict")
                    else event
                )
                for event in self.events
            ],

            "audio_duration": round(
                audio_duration,
                3,
            ),

            "speech_duration": round(
                speech_duration,
                3,
            ),

            "frames_processed": (
                self.frames_processed
            ),

            "speech_frames": (
                self.speech_frames
            ),

            "silence_frames": (
                self.silence_frames
            ),

            "processing_time": round(
                max(
                    0.0,
                    self.processing_time,
                ),
                6,
            ),

            "created_at": self.created_at,
        }


# ====================================================================
# PIPELINE
# ====================================================================

class DiarizationPipeline:
    """
    Pipeline principal de diarisation SpeakerIdentity IA.
    """

    def __init__(
        self,
        session_manager: Optional[
            DiarizationSessionManager
        ] = None,
        audio_processor: Optional[
            AudioProcessor
        ] = None,
        vad: Optional[
            VoiceActivityDetector
        ] = None,
        segment_manager: Optional[
            SegmentManager
        ] = None,
        embedding_engine: Optional[
            EmbeddingEngine
        ] = None,
        speaker_manager: Optional[
            SpeakerManager
        ] = None,
        metrics: Optional[
            DiarizationMetrics
        ] = None,
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

        self.session_manager = (
            session_manager
            if session_manager is not None
            else DiarizationSessionManager(
                config=self.config
            )
        )

        self.audio_processor = (
            audio_processor
            if audio_processor is not None
            else AudioProcessor(
                config=self.config
            )
        )

        self.vad = (
            vad
            if vad is not None
            else VoiceActivityDetector(
                config=self.config
            )
        )

        self.segment_manager = (
            segment_manager
            if segment_manager is not None
            else SegmentManager(
                config=self.config
            )
        )

        self.embedding_engine = (
            embedding_engine
            if embedding_engine is not None
            else EmbeddingEngine(
                config=self.config
            )
        )

        self.speaker_manager = (
            speaker_manager
            if speaker_manager is not None
            else SpeakerManager(
                config=self.config
            )
        )

        self.metrics = (
            metrics
            if metrics is not None
            else default_metrics
        )

        self._session_audio_offsets: Dict[
            str,
            float,
        ] = {}

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

        try:

            session = (
                self.session_manager.create_session(
                    session_id=session_id,
                    metadata=metadata,
                )
            )

            if session is None:

                raise SessionError(
                    "Session manager returned no session."
                )

            self.metrics.session_created(
                session.session_id
            )

            self.session_manager.start_session(
                session.session_id
            )

            self.metrics.session_started(
                session.session_id
            )

            self._session_audio_offsets[
                session.session_id
            ] = 0.0

            logger.debug(
                "Diarization session started: %s",
                session.session_id,
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

            raise PipelineError(
                f"Unable to start session: {exc}"
            ) from exc

    # =================================================================
    # STOP SESSION
    # =================================================================

    def stop_session(
        self,
        session_id: str,
    ) -> Any:

        if not isinstance(
            session_id,
            str,
        ) or not session_id.strip():

            raise SessionError(
                "session_id is required."
            )

        session_id = session_id.strip()

        try:

            session = (
                self.session_manager.stop_session(
                    session_id
                )
            )

            self.metrics.session_stopped(
                session_id
            )

            self._session_audio_offsets.pop(
                session_id,
                None,
            )

            logger.debug(
                "Diarization session stopped: %s",
                session_id,
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
                "errors",
                session_id=session_id,
            )

            raise PipelineError(
                f"Unable to stop session "
                f"{session_id}: {exc}"
            ) from exc

    # =================================================================
    # PROCESS
    # =================================================================

    def process(
        self,
        session_id: str,
        audio: bytes,
    ) -> PipelineResult:

        if not isinstance(
            session_id,
            str,
        ) or not session_id.strip():

            raise SessionError(
                "session_id is required."
            )

        session_id = session_id.strip()

        if audio is None:

            raise AudioFormatError(
                "Audio cannot be None."
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
                "Pipeline expects PCM16 bytes."
            )

        audio = bytes(audio)

        start_time = time.perf_counter()

        events: List[
            DiarizationEvent
        ] = []

        segments: List[Any] = []

        frames_processed = 0
        speech_frames = 0
        silence_frames = 0

        speech_duration = 0.0

        try:

            # =========================================================
            # SESSION
            # =========================================================

            session = (
                self.session_manager.get_session(
                    session_id
                )
            )

            if session is None:

                raise SessionNotFoundError(
                    f"Session not found: {session_id}"
                )

            # =========================================================
            # EMPTY AUDIO
            # =========================================================

            if not audio:

                return PipelineResult(
                    session_id=session_id,
                    segments=[],
                    events=[],
                )

            # =========================================================
            # PCM16 ALIGNMENT
            # =========================================================

            if (
                len(audio)
                % self.config.sample_width
                != 0
            ):

                raise AudioFormatError(
                    "PCM16 audio payload has an "
                    "invalid byte length."
                )

            # =========================================================
            # AUDIO PROCESSOR
            # =========================================================

            processed_audio = (
                self.audio_processor.process(
                    audio
                )
            )

            if processed_audio is None:

                processed_audio = audio

            if not isinstance(
                processed_audio,
                (
                    bytes,
                    bytearray,
                    memoryview,
                ),
            ):

                raise AudioFormatError(
                    "AudioProcessor must return "
                    "PCM16 bytes."
                )

            processed_audio = bytes(
                processed_audio
            )

            if not processed_audio:

                return PipelineResult(
                    session_id=session_id,
                    segments=[],
                    events=[],
                )

            if (
                len(processed_audio)
                % self.config.sample_width
                != 0
            ):

                raise AudioFormatError(
                    "Processed PCM16 audio payload "
                    "has an invalid byte length."
                )

            # =========================================================
            # AUDIO DURATION
            # =========================================================

            bytes_per_second = (
                self.config.sample_rate
                * self.config.sample_width
                * self.config.channels
            )

            if bytes_per_second <= 0:

                raise AudioFormatError(
                    "Invalid audio configuration."
                )

            audio_duration = (
                len(processed_audio)
                / bytes_per_second
            )

            audio_duration = max(
                0.0,
                audio_duration,
            )

            # =========================================================
            # METRICS
            # =========================================================

            self.metrics.increment(
                "audio_chunks",
                session_id=session_id,
            )

            self.metrics.increment(
                "audio_bytes",
                len(processed_audio),
                session_id=session_id,
            )

            # =========================================================
            # FRAMES
            # =========================================================

            frames = (
                self.audio_processor.iter_frames(
                    processed_audio
                )
            )

            if frames is None:

                frames = []

            frame_timestamp = (
                self._session_audio_offsets.get(
                    session_id,
                    0.0,
                )
            )

            # =========================================================
            # FRAME PROCESSING
            # =========================================================

            for frame in frames:

                if frame is None:
                    continue

                if not isinstance(
                    frame,
                    (
                        bytes,
                        bytearray,
                        memoryview,
                    ),
                ):

                    raise AudioFormatError(
                        "AudioProcessor returned "
                        "an invalid frame type."
                    )

                frame = bytes(frame)

                if not frame:
                    continue

                if (
                    len(frame)
                    % self.config.sample_width
                    != 0
                ):

                    raise AudioFormatError(
                        "Audio frame has an "
                        "invalid PCM16 byte length."
                    )

                frame_duration = (
                    len(frame)
                    / bytes_per_second
                )

                frame_duration = max(
                    0.0,
                    frame_duration,
                )

                expected_vad_size = (
                    self.vad.frame_bytes
                )

                if len(frame) != expected_vad_size:

                    raise AudioFormatError(
                        "Audio frame size is "
                        "incompatible with VAD: "
                        f"expected "
                        f"{expected_vad_size} bytes, "
                        f"got {len(frame)}."
                    )

                # -----------------------------------------------------
                # VAD
                # -----------------------------------------------------

                vad_result = (
                    self.vad.process_frame(
                        frame,
                        timestamp=frame_timestamp,
                    )
                )

                is_speech = (
                    self._vad_is_speech(
                        vad_result
                    )
                )

                frames_processed += 1

                if is_speech:

                    speech_frames += 1

                    speech_duration += (
                        frame_duration
                    )

                    self.metrics.increment(
                        "speech_frames",
                        session_id=session_id,
                    )

                else:

                    silence_frames += 1

                    self.metrics.increment(
                        "silence_frames",
                        session_id=session_id,
                    )

                self.metrics.increment(
                    "frames_processed",
                    session_id=session_id,
                )

                # -----------------------------------------------------
                # SEGMENT
                # -----------------------------------------------------

                generated_segments = (
                    self._process_segment(
                        session_id=session_id,
                        frame=frame,
                        is_speech=is_speech,
                        vad_result=vad_result,
                    )
                )

                if generated_segments:

                    segments.extend(
                        generated_segments
                    )

                frame_timestamp += (
                    frame_duration
                )

            # =========================================================
            # UPDATE TIMESTAMP
            # =========================================================

            self._session_audio_offsets[
                session_id
            ] = (
                self._session_audio_offsets.get(
                    session_id,
                    0.0,
                )
                + audio_duration
            )

            speech_duration = min(
                max(
                    0.0,
                    speech_duration,
                ),
                audio_duration,
            )

            # =========================================================
            # SPEAKER IDENTIFICATION
            # =========================================================

            for segment in segments:

                self._identify_segment_speaker(
                    session_id=session_id,
                    segment=segment,
                    events=events,
                )

            # =========================================================
            # SESSION STATISTICS
            # =========================================================

            self.session_manager.update_audio_stats(
                session_id=session_id,
                duration=audio_duration,
                speech_duration=speech_duration,
            )

            # =========================================================
            # PROCESSING TIME
            # =========================================================

            processing_time = (
                time.perf_counter()
                - start_time
            )

            self.metrics.add_processing_time(
                processing_time,
                session_id=session_id,
            )

            return PipelineResult(
                session_id=session_id,
                segments=segments,
                events=events,
                audio_duration=audio_duration,
                speech_duration=speech_duration,
                frames_processed=frames_processed,
                speech_frames=speech_frames,
                silence_frames=silence_frames,
                processing_time=processing_time,
            )

        except (
            DiarizationError,
            SessionError,
            SessionNotFoundError,
            AudioFormatError,
        ):

            self.metrics.increment(
                "errors",
                session_id=session_id,
            )

            raise

        except Exception as exc:

            self.metrics.increment(
                "errors",
                session_id=session_id,
            )

            raise PipelineError(
                f"Audio pipeline failed for "
                f"session {session_id}: {exc}"
            ) from exc

    # =================================================================
    # VAD NORMALIZATION
    # =================================================================

    @staticmethod
    def _vad_is_speech(
        vad_result: Any,
    ) -> bool:

        if isinstance(
            vad_result,
            bool,
        ):

            return vad_result

        if isinstance(
            vad_result,
            dict,
        ):

            if "is_speech" in vad_result:

                return bool(
                    vad_result["is_speech"]
                )

            if "speech" in vad_result:

                return bool(
                    vad_result["speech"]
                )

            return False

        if hasattr(
            vad_result,
            "is_speech",
        ):

            return bool(
                vad_result.is_speech
            )

        if hasattr(
            vad_result,
            "speech",
        ):

            return bool(
                vad_result.speech
            )

        return False

    # =================================================================
    # SEGMENT
    # =================================================================

    def _process_segment(
        self,
        session_id: str,
        frame: bytes,
        is_speech: bool,
        vad_result: Any,
    ) -> List[Any]:

        try:

            method = getattr(
                self.segment_manager,
                "process_frame",
                None,
            )

            if callable(method):

                try:

                    result = method(
                        session_id=session_id,
                        frame=frame,
                        is_speech=is_speech,
                        vad_result=vad_result,
                    )

                except TypeError:

                    result = method(
                        session_id,
                        frame,
                        is_speech,
                    )

            else:

                method = getattr(
                    self.segment_manager,
                    "process",
                    None,
                )

                if not callable(method):

                    return []

                try:

                    result = method(
                        session_id=session_id,
                        frame=frame,
                        is_speech=is_speech,
                        vad_result=vad_result,
                    )

                except TypeError:

                    result = method(
                        session_id=session_id,
                        frame=frame,
                        is_speech=is_speech,
                    )

            if result is None:
                return []

            if isinstance(
                result,
                (list, tuple),
            ):

                return list(result)

            return [result]

        except (
            SessionError,
            DiarizationError,
        ):

            raise

        except Exception as exc:

            raise PipelineError(
                f"Segment processing failed: {exc}"
            ) from exc

    # =================================================================
    # SPEAKER IDENTIFICATION
    # =================================================================

    def _identify_segment_speaker(
        self,
        session_id: str,
        segment: Any,
        events: List[DiarizationEvent],
    ) -> None:

        if not self.config.embedding_enabled:
            return

        if not self.config.speaker_identification_enabled:
            return

        audio_data = (
            self._extract_segment_audio(
                segment
            )
        )

        if not audio_data:

            logger.debug(
                "Segment contains no audio data; "
                "speaker identification skipped."
            )

            return

        # =============================================================
        # EMBEDDING
        # =============================================================

        try:

            embedding = (
                self.embedding_engine.embed(
                    audio_data
                )
            )

        except EmbeddingError as exc:

            message = str(exc).lower()

            is_short_segment = (
                "too short" in message
                or (
                    "short" in message
                    and "embedding" in message
                )
            )

            if is_short_segment:

                logger.debug(
                    "Segment ignored for embedding "
                    "because it is too short."
                )

                return

            is_long_segment = (
                "too long" in message
                or (
                    "long" in message
                    and "embedding" in message
                )
            )

            if not is_long_segment:
                raise

            embedding_audio = (
                self._limit_embedding_audio(
                    audio_data
                )
            )

            if not embedding_audio:

                logger.debug(
                    "Segment contains no compatible "
                    "audio for embedding."
                )

                return

            try:

                embedding = (
                    self.embedding_engine.embed(
                        embedding_audio
                    )
                )

            except EmbeddingError as retry_exc:

                retry_message = (
                    str(retry_exc).lower()
                )

                if (
                    "too short" in retry_message
                    or (
                        "short" in retry_message
                        and "embedding"
                        in retry_message
                    )
                ):

                    logger.debug(
                        "Limited segment is too short "
                        "for speaker embedding."
                    )

                    return

                raise

        if embedding is None:

            logger.debug(
                "Embedding engine returned no embedding."
            )

            return

        # =============================================================
        # VECTEUR
        # =============================================================
        #
        # CORRECTIF : EmbeddingEngine.embed() renvoie un EmbeddingResult
        # (un dataclass), pas le vecteur. Le pipeline transmettait cet
        # objet tel quel a SpeakerManager.identify(), qui appelle
        # np.asarray(...) dessus et echoue :
        #
        #     DiarizationError: Invalid speaker embedding.
        #
        # L'exception se produisait sur CHAQUE segment. Aucun locuteur
        # ne pouvait donc jamais etre identifie, quelle que soit la
        # qualite de l'audio ou du modele.
        #
        # On extrait desormais le vecteur, tout en acceptant qu'un
        # moteur renvoie directement une sequence de nombres.

        vecteur = getattr(
            embedding,
            "embedding",
            embedding,
        )

        # =============================================================
        # SPEAKER MANAGER
        # =============================================================

        result = self._identify_speaker(
            session_id=session_id,
            embedding=vecteur,
        )

        speaker_id = None
        confidence = None
        created = False
        similarity = None

        if isinstance(
            result,
            str,
        ):

            speaker_id = result

        elif isinstance(
            result,
            dict,
        ):

            speaker_id = result.get(
                "speaker_id"
            )

            confidence = result.get(
                "confidence"
            )

            similarity = result.get(
                "similarity"
            )

            created = bool(
                result.get(
                    "created",
                    False,
                )
            )

        else:

            speaker_id = getattr(
                result,
                "speaker_id",
                None,
            )

            confidence = getattr(
                result,
                "confidence",
                None,
            )

            similarity = getattr(
                result,
                "similarity",
                None,
            )

            created = bool(
                getattr(
                    result,
                    "created",
                    False,
                )
            )

        if not speaker_id:

            logger.debug(
                "SpeakerManager returned no speaker "
                "for session=%s.",
                session_id,
            )

            return

        # =============================================================
        # ASSIGN SPEAKER
        # =============================================================

        if isinstance(
            segment,
            dict,
        ):

            segment["speaker_id"] = speaker_id

            segment["is_new_speaker"] = bool(created)

            if similarity is not None:

                segment["speaker_similarity"] = similarity

            if confidence is not None:

                segment["speaker_confidence"] = confidence

        else:

            # CORRECTIF : l'affectation etait gardee par
            #
            #     if hasattr(segment, "speaker_id")
            #
            # or SegmentResult n'avait pas ce champ : le garde etait
            # donc toujours faux et l'identifiant n'etait JAMAIS pose
            # sur le segment. Il ne vivait que dans les evenements.
            # Le navigateur, qui lit segment.speaker_id, affichait
            # "speaker_unknown" pour tous les segments, meme quand
            # l'identification avait parfaitement fonctionne.
            #
            # Le champ existe maintenant sur SegmentResult, et on
            # l'ecrit sans condition.

            segment.speaker_id = speaker_id

            segment.is_new_speaker = bool(created)

            if similarity is not None:

                try:
                    segment.speaker_similarity = float(
                        similarity
                    )
                except (TypeError, ValueError):
                    segment.speaker_similarity = 0.0

            try:
                segment.speaker_threshold = float(
                    self.speaker_manager._get_similarity_threshold()
                )
            except Exception:
                segment.speaker_threshold = 0.0

            if confidence is not None:

                try:
                    segment.speaker_confidence = float(
                        confidence
                    )
                except (TypeError, ValueError):
                    segment.speaker_confidence = 0.0

            # segment.confidence reste la confiance du VAD : on ne
            # l'ecrase plus avec celle de l'identification, qui ne
            # mesure pas la meme chose.

        self.session_manager.assign_speaker(
            session_id=session_id,
            speaker_id=speaker_id,
            segment=segment,
        )

        self.metrics.increment(
            "speakers_identified",
            session_id=session_id,
        )

        # =============================================================
        # SEGMENT INFORMATION
        # =============================================================

        segment_id = getattr(
            segment,
            "segment_id",
            None,
        )

        start = getattr(
            segment,
            "start",
            None,
        )

        end = getattr(
            segment,
            "end",
            None,
        )

        if isinstance(
            segment,
            dict,
        ):

            segment_id = segment.get(
                "segment_id",
                segment_id,
            )

            start = segment.get(
                "start",
                start,
            )

            end = segment.get(
                "end",
                end,
            )

        speaker_event_data = {
            "speaker_id": speaker_id,
            "confidence": confidence,
        }

        # =============================================================
        # SPEAKER EVENT
        # =============================================================

        if created:

            self.metrics.increment(
                "speakers_created",
                session_id=session_id,
            )

            events.append(
                DiarizationEvent(
                    event_type=(
                        EventTypes.SPEAKER_CREATED
                    ),
                    session_id=session_id,
                    data=speaker_event_data,
                )
            )

        else:

            events.append(
                DiarizationEvent(
                    event_type=(
                        EventTypes.SPEAKER_IDENTIFIED
                    ),
                    session_id=session_id,
                    data=speaker_event_data,
                )
            )

        # =============================================================
        # SEGMENT EVENT
        # =============================================================

        events.append(
            DiarizationEvent(
                event_type=(
                    EventTypes.SEGMENT_CREATED
                ),
                session_id=session_id,
                data={
                    "speaker_id": speaker_id,
                    "segment_id": segment_id,
                    "start": start,
                    "end": end,
                    "confidence": confidence,
                },
            )
        )

    # =================================================================
    # LIMIT EMBEDDING AUDIO
    # =================================================================

    def _limit_embedding_audio(
        self,
        audio_data: bytes,
    ) -> bytes:

        if not audio_data:
            return b""

        sample_rate = int(
            getattr(
                self.config,
                "sample_rate",
                16000,
            )
        )

        sample_width = int(
            getattr(
                self.config,
                "sample_width",
                2,
            )
        )

        channels = int(
            getattr(
                self.config,
                "channels",
                1,
            )
        )

        if (
            sample_rate <= 0
            or sample_width <= 0
            or channels <= 0
        ):

            raise AudioFormatError(
                "Invalid audio configuration "
                "for embedding."
            )

        bytes_per_second = (
            sample_rate
            * sample_width
            * channels
        )

        max_duration_seconds = (
            EMBEDDING_MAX_DURATION_MS
            / 1000.0
        )

        max_bytes = int(
            bytes_per_second
            * max_duration_seconds
        )

        frame_width = (
            sample_width
            * channels
        )

        if frame_width <= 0:

            raise AudioFormatError(
                "Invalid PCM frame width."
            )

        max_bytes -= (
            max_bytes % frame_width
        )

        if max_bytes <= 0:
            return b""

        if len(audio_data) <= max_bytes:
            return audio_data

        limited_audio = (
            audio_data[:max_bytes]
        )

        logger.debug(
            "Embedding audio limited from %.3f ms "
            "to %.3f ms.",
            (
                len(audio_data)
                / bytes_per_second
                * 1000.0
            ),
            (
                len(limited_audio)
                / bytes_per_second
                * 1000.0
            ),
        )

        return limited_audio

    # =================================================================
    # IDENTIFY SPEAKER
    # =================================================================

    def _identify_speaker(
        self,
        session_id: str,
        embedding: Any,
    ) -> Any:
        """
        Identifie un speaker à partir de son embedding.

        Compatibilité avec les signatures existantes :

            identify(
                session_id=session_id,
                embedding=embedding
            )

        ou :

            identify(
                embedding,
                session_id
            )

        ou :

            identify_or_create(
                session_id=session_id,
                embedding=embedding
            )

        Le pipeline ne crée pas lui-même les speakers.
        Cette responsabilité reste dans SpeakerManager.
        """

        # =============================================================
        # IDENTIFY
        # =============================================================

        method = getattr(
            self.speaker_manager,
            "identify",
            None,
        )

        if callable(method):

            try:

                return method(
                    session_id=session_id,
                    embedding=embedding,
                )

            except TypeError as first_error:

                try:

                    return method(
                        embedding,
                        session_id,
                    )

                except TypeError:

                    logger.debug(
                        "SpeakerManager.identify() "
                        "signature is incompatible.",
                        exc_info=True,
                    )

                    raise PipelineError(
                        "SpeakerManager.identify() "
                        "has an unsupported signature."
                    ) from first_error

        # =============================================================
        # IDENTIFY OR CREATE
        # =============================================================

        method = getattr(
            self.speaker_manager,
            "identify_or_create",
            None,
        )

        if callable(method):

            try:

                return method(
                    session_id=session_id,
                    embedding=embedding,
                )

            except TypeError as first_error:

                try:

                    return method(
                        embedding,
                        session_id,
                    )

                except TypeError:

                    logger.debug(
                        "SpeakerManager.identify_or_create() "
                        "signature is incompatible.",
                        exc_info=True,
                    )

                    raise PipelineError(
                        "SpeakerManager.identify_or_create() "
                        "has an unsupported signature."
                    ) from first_error

        raise PipelineError(
            "SpeakerManager does not expose "
            "identify() or identify_or_create()."
        )

    # =================================================================
    # SEGMENT AUDIO
    # =================================================================

    @staticmethod
    def _extract_segment_audio(
        segment: Any,
    ) -> Optional[bytes]:

        if isinstance(
            segment,
            dict,
        ):

            for key in (
                "audio",
                "audio_data",
                "pcm_data",
                "data",
            ):

                value = segment.get(
                    key
                )

                if isinstance(
                    value,
                    (
                        bytes,
                        bytearray,
                        memoryview,
                    ),
                ):

                    return bytes(
                        value
                    )

            return None

        for attribute in (
            "audio",
            "audio_data",
            "pcm_data",
            "data",
        ):

            value = getattr(
                segment,
                attribute,
                None,
            )

            if isinstance(
                value,
                (
                    bytes,
                    bytearray,
                    memoryview,
                ),
            ):

                return bytes(
                    value
                )

        return None

    # =================================================================
    # FLUSH
    # =================================================================

    def flush(
        self,
        session_id: str,
    ) -> PipelineResult:

        if not isinstance(
            session_id,
            str,
        ) or not session_id.strip():

            raise SessionError(
                "session_id is required."
            )

        session_id = session_id.strip()

        start_time = time.perf_counter()

        events: List[
            DiarizationEvent
        ] = []

        segments: List[Any] = []

        try:

            session = (
                self.session_manager.get_session(
                    session_id
                )
            )

            if session is None:

                raise SessionNotFoundError(
                    f"Session not found: {session_id}"
                )

            method = getattr(
                self.segment_manager,
                "flush",
                None,
            )

            if callable(method):

                try:

                    result = method(
                        session_id
                    )

                except TypeError:

                    result = method()

                if result is not None:

                    if isinstance(
                        result,
                        (list, tuple),
                    ):

                        segments.extend(
                            result
                        )

                    else:

                        segments.append(
                            result
                        )

            for segment in segments:

                self._identify_segment_speaker(
                    session_id=session_id,
                    segment=segment,
                    events=events,
                )

            processing_time = (
                time.perf_counter()
                - start_time
            )

            self.metrics.add_processing_time(
                processing_time,
                session_id=session_id,
            )

            return PipelineResult(
                session_id=session_id,
                segments=segments,
                events=events,
                processing_time=processing_time,
            )

        except (
            SessionNotFoundError,
            SessionError,
            DiarizationError,
        ):

            self.metrics.increment(
                "errors",
                session_id=session_id,
            )

            raise

        except Exception as exc:

            self.metrics.increment(
                "errors",
                session_id=session_id,
            )

            raise PipelineError(
                f"Unable to flush session "
                f"{session_id}: {exc}"
            ) from exc

    # =================================================================
    # HEALTH
    # =================================================================

    def health(
        self,
    ) -> Dict[str, Any]:

        components: Dict[
            str,
            Any,
        ] = {}

        for name, component in (
            (
                "audio_processor",
                self.audio_processor,
            ),
            (
                "vad",
                self.vad,
            ),
            (
                "segment_manager",
                self.segment_manager,
            ),
            (
                "embedding_engine",
                self.embedding_engine,
            ),
            (
                "speaker_manager",
                self.speaker_manager,
            ),
            (
                "session_manager",
                self.session_manager,
            ),
        ):

            try:

                method = getattr(
                    component,
                    "health",
                    None,
                )

                if callable(method):

                    components[name] = method()

                else:

                    components[name] = {
                        "status": "healthy"
                    }

            except Exception as exc:

                components[name] = {
                    "status": "unhealthy",
                    "error": str(exc),
                }

        overall = "healthy"

        for component in components.values():

            if not isinstance(
                component,
                dict,
            ):

                overall = "degraded"
                continue

            status = component.get(
                "status"
            )

            if status not in (
                None,
                "healthy",
            ):

                overall = "degraded"

        return {
            "service":
                "speakeridentity_diarization_pipeline",

            "status":
                overall,

            "components":
                components,
        }

    # =================================================================
    # STATISTICS
    # =================================================================

    def statistics(
        self,
    ) -> Dict[str, Any]:

        return self.metrics.snapshot_dict()


# ====================================================================
# FACTORY
# ====================================================================

def create_pipeline(
    config: Optional[
        DiarizationConfig
    ] = None,
) -> DiarizationPipeline:

    return DiarizationPipeline(
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
    "PipelineResult",
    "DiarizationPipeline",
    "Pipeline",
    "create_pipeline",
]


# ====================================================================
# ALIAS
# ====================================================================
#
# Les tests et le serveur de demonstration utilisent le nom court
# Pipeline. La classe reelle s'appelle DiarizationPipeline. On expose
# les deux plutot que de renommer la classe.

Pipeline = DiarizationPipeline