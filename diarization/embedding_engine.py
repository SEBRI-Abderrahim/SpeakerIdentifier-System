"""
====================================================================
SpeakerIdentity IA - Diarization Embedding Engine
====================================================================

Projet       : SpeakerIdentity IA
Module       : Diarization
Fichier      : diarization/embedding_engine.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 2.2.7
Date         : 15/08/2026

Description
-----------

Moteur de génération et de comparaison d'embeddings vocaux.

Format interne SpeakerIdentity:

    PCM16 / 16000 Hz / Mono

Le modèle SpeechBrain ECAPA est chargé une seule fois
par processus Python.

Le modèle partagé est utilisé par :

    - toutes les instances EmbeddingEngine ;
    - toutes les sessions ;
    - tous les speakers ;
    - tous les appels embed().

L'identification du speaker reste assurée par :

    - speaker_manager.py
    - speaker_memory.py

====================================================================
"""

from __future__ import annotations

import logging
import math
import threading
import time

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional


from .config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_ENABLED,
    EMBEDDING_MAX_AUDIO_MS,
    EMBEDDING_MIN_AUDIO_MS,
    EMBEDDING_MODEL,
    EMBEDDING_SAMPLE_RATE,
    CPU_THREADS,
    DEVICE,
    GPU_DEVICE_INDEX,
    INFERENCE_BATCH_SIZE,
    USE_GPU,
    SAMPLE_RATE,
    CHANNELS,
    SAMPLE_WIDTH,
)

from .exceptions import (
    AudioFormatError,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingModelError,
)

from .metrics import metrics


logger = logging.getLogger(
    "speakeridentity.diarization.embedding"
)


# ====================================================================
# OPTIONAL DEPENDENCIES
# ====================================================================

try:
    import numpy as np

except ImportError:  # pragma: no cover
    np = None


try:
    import torch

except ImportError:  # pragma: no cover
    torch = None


try:
    from speechbrain.inference.speaker import EncoderClassifier

except ImportError:  # pragma: no cover

    try:
        from speechbrain.pretrained import EncoderClassifier

    except ImportError:  # pragma: no cover
        EncoderClassifier = None


# ====================================================================
# EMBEDDING RESULT
# ====================================================================

@dataclass(frozen=True)
class EmbeddingResult:
    """
    Résultat d'un calcul d'embedding.
    """

    embedding: tuple[float, ...]

    dimension: int

    sample_rate: int

    duration_seconds: float

    device: str

    model: str

    processing_time: float

    normalized: bool = True

    def to_list(self) -> list[float]:
        """
        Retourne l'embedding sous forme de liste.
        """

        return list(
            self.embedding
        )

    def to_dict(
        self,
        include_embedding: bool = False,
    ) -> dict[str, Any]:
        """
        Convertit le résultat en dictionnaire.
        """

        result: dict[str, Any] = {
            "dimension": self.dimension,
            "sample_rate": self.sample_rate,
            "duration_seconds": round(
                max(
                    0.0,
                    self.duration_seconds,
                ),
                4,
            ),
            "device": self.device,
            "model": self.model,
            "processing_time": round(
                max(
                    0.0,
                    self.processing_time,
                ),
                6,
            ),
            "normalized": self.normalized,
        }

        if include_embedding:
            result["embedding"] = self.to_list()

        return result


# ====================================================================
# EMBEDDING ENGINE
# ====================================================================

class EmbeddingEngine:
    """
    Moteur central d'embeddings vocaux SpeakerIdentity.

    Le modèle SpeechBrain ECAPA est partagé globalement
    au niveau du processus Python.
    """

    # ================================================================
    # SHARED MODEL
    # ================================================================

    _shared_model: Any = None

    _shared_loaded: bool = False

    _shared_load_error: Optional[str] = None

    _shared_model_name: Optional[str] = None

    _shared_device: Optional[str] = None

    _shared_torch_device: Optional[str] = None

    _shared_lock = threading.RLock()

    _shared_load_count: int = 0

    # ================================================================
    # CONSTRUCTOR
    # ================================================================

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: Optional[str] = None,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        sample_rate: int = EMBEDDING_SAMPLE_RATE,
        min_audio_ms: int = EMBEDDING_MIN_AUDIO_MS,
        max_audio_ms: int = EMBEDDING_MAX_AUDIO_MS,
        enabled: bool = EMBEDDING_ENABLED,
        gpu_device_index: int = GPU_DEVICE_INDEX,
        cpu_threads: int = CPU_THREADS,
        inference_batch_size: int = INFERENCE_BATCH_SIZE,
        auto_load: bool = False,
        config: Optional[Any] = None,
    ) -> None:

        self.config = config

        # ------------------------------------------------------------
        # CONFIGURATION
        # ------------------------------------------------------------

        model_name = self._config_value(
            config,
            (
                "embedding_model",
                "EMBEDDING_MODEL",
                "model_name",
            ),
            model_name,
        )

        configured_device = self._config_value(
            config,
            (
                "embedding_device",
                "EMBEDDING_DEVICE",
                "device",
            ),
            device,
        )

        configured_use_gpu = self._config_value(
            config,
            (
                "use_gpu",
                "USE_GPU",
            ),
            USE_GPU,
        )

        embedding_dimension = self._config_value(
            config,
            (
                "embedding_dimension",
                "EMBEDDING_DIMENSION",
            ),
            embedding_dimension,
        )

        sample_rate = self._config_value(
            config,
            (
                "embedding_sample_rate",
                "EMBEDDING_SAMPLE_RATE",
            ),
            sample_rate,
        )

        min_audio_ms = self._config_value(
            config,
            (
                "embedding_min_audio_ms",
                "EMBEDDING_MIN_AUDIO_MS",
                "min_audio_ms",
            ),
            min_audio_ms,
        )

        max_audio_ms = self._config_value(
            config,
            (
                "embedding_max_audio_ms",
                "EMBEDDING_MAX_AUDIO_MS",
                "max_audio_ms",
            ),
            max_audio_ms,
        )

        enabled = self._config_value(
            config,
            (
                "embedding_enabled",
                "EMBEDDING_ENABLED",
                "enabled",
            ),
            enabled,
        )

        gpu_device_index = self._config_value(
            config,
            (
                "gpu_device_index",
                "GPU_DEVICE_INDEX",
            ),
            gpu_device_index,
        )

        cpu_threads = self._config_value(
            config,
            (
                "cpu_threads",
                "CPU_THREADS",
            ),
            cpu_threads,
        )

        inference_batch_size = self._config_value(
            config,
            (
                "inference_batch_size",
                "INFERENCE_BATCH_SIZE",
            ),
            inference_batch_size,
        )

        self.model_name = str(
            model_name
        ).strip()

        self.embedding_dimension = (
            self._positive_int(
                embedding_dimension,
                "Embedding dimension",
                EmbeddingDimensionError,
            )
        )

        self.sample_rate = (
            self._positive_int(
                sample_rate,
                "Embedding sample rate",
                EmbeddingError,
            )
        )

        self.min_audio_ms = (
            self._positive_int(
                min_audio_ms,
                "Minimum embedding audio duration",
                EmbeddingError,
            )
        )

        self.max_audio_ms = (
            self._positive_int(
                max_audio_ms,
                "Maximum embedding audio duration",
                EmbeddingError,
            )
        )

        self.enabled = self._to_bool(
            enabled
        )

        self.gpu_device_index = (
            self._non_negative_int(
                gpu_device_index,
                "gpu_device_index",
            )
        )

        self.cpu_threads = max(
            1,
            self._integer(
                cpu_threads,
                "cpu_threads",
            ),
        )

        self.inference_batch_size = max(
            1,
            self._integer(
                inference_batch_size,
                "inference_batch_size",
            ),
        )

        self.use_gpu = self._to_bool(
            configured_use_gpu
        )

        # ------------------------------------------------------------
        # INSTANCE STATE
        # ------------------------------------------------------------

        self._lock = (
            EmbeddingEngine._shared_lock
        )

        self._inference_count = 0

        self._total_processing_time = 0.0

        self._created_at = time.time()

        self.device = self._resolve_device(
            configured_device
        )

        self._validate_configuration()

        if auto_load:
            self.load()

    # ================================================================
    # UTILITIES
    # ================================================================

    @staticmethod
    def _integer(
        value: Any,
        name: str,
    ) -> int:

        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise EmbeddingError(
                f"{name} must be an integer."
            ) from exc

    @classmethod
    def _positive_int(
        cls,
        value: Any,
        name: str,
        error_type: type[Exception],
    ) -> int:

        try:
            result = int(value)

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise error_type(
                f"{name} must be an integer."
            ) from exc

        if result <= 0:
            raise error_type(
                f"{name} must be positive."
            )

        return result

    @classmethod
    def _non_negative_int(
        cls,
        value: Any,
        name: str,
    ) -> int:

        result = cls._integer(
            value,
            name,
        )

        if result < 0:
            raise EmbeddingError(
                f"{name} must be >= 0."
            )

        return result

    @staticmethod
    def _to_bool(
        value: Any,
    ) -> bool:

        if isinstance(
            value,
            bool,
        ):
            return value

        if isinstance(
            value,
            str,
        ):

            normalized = (
                value
                .strip()
                .lower()
            )

            if normalized in {
                "1",
                "true",
                "yes",
                "y",
                "on",
                "enabled",
            }:
                return True

            if normalized in {
                "0",
                "false",
                "no",
                "n",
                "off",
                "disabled",
            }:
                return False

        return bool(value)

    @staticmethod
    def _config_value(
        config: Optional[Any],
        names: Iterable[str],
        default: Any,
    ) -> Any:

        if config is None:
            return default

        for name in names:

            if (
                isinstance(
                    config,
                    Mapping,
                )
                and name in config
            ):

                value = config[name]

                if value is not None:
                    return value

            try:

                value = getattr(
                    config,
                    name,
                )

            except AttributeError:
                continue

            if value is not None:
                return value

        return default

    # ================================================================
    # DEVICE
    # ================================================================

    def _resolve_device(
        self,
        requested_device: Optional[str],
    ) -> str:

        requested = (
            requested_device
            if requested_device is not None
            else DEVICE
        )

        requested = str(
            requested
        ).strip().lower()

        if requested in {
            "cpu",
            "cpu:0",
        }:

            return "cpu"

        if requested.startswith(
            "cuda"
        ):

            if (
                torch is not None
                and torch.cuda.is_available()
            ):

                return "cuda"

            logger.warning(
                "CUDA requested but unavailable. "
                "Falling back to CPU."
            )

            return "cpu"

        if requested in {
            "auto",
            "",
        }:

            if (
                self.use_gpu
                and torch is not None
                and torch.cuda.is_available()
            ):

                return "cuda"

            return "cpu"

        logger.warning(
            "Unknown embedding device '%s'. "
            "Falling back to CPU.",
            requested,
        )

        return "cpu"

    @property
    def torch_device(self) -> str:

        if self.device == "cuda":

            return (
                f"cuda:{self.gpu_device_index}"
            )

        return "cpu"

    # ================================================================
    # VALIDATION
    # ================================================================

    def _validate_configuration(
        self,
    ) -> None:

        if not self.model_name:

            raise EmbeddingError(
                "Embedding model name cannot be empty."
            )

        if self.sample_rate != SAMPLE_RATE:

            raise EmbeddingError(
                "SpeakerIdentity diarization requires "
                f"{SAMPLE_RATE} Hz audio."
            )

        if self.max_audio_ms < self.min_audio_ms:

            raise EmbeddingError(
                "max_audio_ms must be greater than "
                "or equal to min_audio_ms."
            )

        if SAMPLE_WIDTH != 2:

            raise EmbeddingError(
                "SpeakerIdentity diarization requires PCM16 "
                "(2 bytes per sample)."
            )

        if CHANNELS != 1:

            raise EmbeddingError(
                "SpeakerIdentity diarization requires mono audio."
            )

    # ================================================================
    # DEPENDENCIES
    # ================================================================

    def _check_dependencies(
        self,
    ) -> None:

        if np is None:

            raise EmbeddingError(
                "NumPy is required for EmbeddingEngine."
            )

        if torch is None:

            raise EmbeddingError(
                "PyTorch is required for EmbeddingEngine."
            )

        if EncoderClassifier is None:

            raise EmbeddingModelError(
                "SpeechBrain is required for "
                "EmbeddingEngine."
            )

    # ================================================================
    # SHARED MODEL LOADING
    # ================================================================

    def load(self) -> None:
        """
        Charge le modèle ECAPA une seule fois.
        """

        if not self.enabled:

            logger.info(
                "EmbeddingEngine is disabled."
            )

            return

        with EmbeddingEngine._shared_lock:

            # --------------------------------------------------------
            # SHARED MODEL ALREADY LOADED
            # --------------------------------------------------------

            if (
                EmbeddingEngine._shared_loaded
                and EmbeddingEngine._shared_model is not None
            ):

                # The shared model belongs to the process.
                # A different model must never silently reuse it.

                if (
                    EmbeddingEngine._shared_model_name
                    != self.model_name
                ):

                    raise EmbeddingModelError(
                        "A shared embedding model is already "
                        "loaded with model "
                        f"'{EmbeddingEngine._shared_model_name}', "
                        "but this instance requested "
                        f"'{self.model_name}'."
                    )

                if (
                    EmbeddingEngine._shared_torch_device
                    != self.torch_device
                ):

                    raise EmbeddingModelError(
                        "A shared embedding model is already "
                        "loaded on "
                        f"'{EmbeddingEngine._shared_torch_device}', "
                        "but this instance requested "
                        f"'{self.torch_device}'."
                    )

                return

            # --------------------------------------------------------
            # DEPENDENCIES
            # --------------------------------------------------------

            run_device = self.torch_device

            try:

                # ----------------------------------------------------
                # DEPENDENCIES
                # ----------------------------------------------------
                #
                # CORRECTIF : cet appel etait place AVANT le try. Quand
                # speechbrain ou torch manquent, l'exception ne passait
                # donc par aucun except : _shared_load_error restait a
                # None et statistics()["load_error"] ne disait rien,
                # alors que c'est precisement la cause de la panne.
                #
                # Il est maintenant a l'interieur du try, pour que la
                # raison exacte soit enregistree et visible en
                # supervision.

                self._check_dependencies()

                # ----------------------------------------------------
                # GPU VALIDATION
                # ----------------------------------------------------

                if self.device == "cuda":

                    if not torch.cuda.is_available():

                        raise EmbeddingModelError(
                            "CUDA became unavailable "
                            "before model loading."
                        )

                    device_count = (
                        torch.cuda.device_count()
                    )

                    if (
                        self.gpu_device_index
                        >= device_count
                    ):

                        raise EmbeddingModelError(
                            "Invalid GPU device index: "
                            f"{self.gpu_device_index}. "
                            "Available GPUs: "
                            f"{device_count}."
                        )

                # ----------------------------------------------------
                # CPU CONFIGURATION
                # ----------------------------------------------------

                if run_device == "cpu":

                    try:

                        torch.set_num_threads(
                            self.cpu_threads
                        )

                        logger.info(
                            "Torch CPU threads configured: %s",
                            self.cpu_threads,
                        )

                    except Exception:

                        logger.debug(
                            "Unable to configure "
                            "Torch CPU threads.",
                            exc_info=True,
                        )

                # ----------------------------------------------------
                # MODEL LOAD
                # ----------------------------------------------------

                logger.info(
                    "Loading shared speaker embedding model: %s",
                    self.model_name,
                )

                logger.info(
                    "Embedding device: %s",
                    run_device,
                )

                model = (
                    EncoderClassifier.from_hparams(
                        source=self.model_name,
                        run_opts={
                            "device": run_device,
                        },
                    )
                )

                if model is None:

                    raise EmbeddingModelError(
                        "SpeechBrain returned an empty "
                        "embedding model."
                    )

                # ----------------------------------------------------
                # SHARED STATE
                # ----------------------------------------------------

                EmbeddingEngine._shared_model = (
                    model
                )

                EmbeddingEngine._shared_loaded = (
                    True
                )

                EmbeddingEngine._shared_load_error = (
                    None
                )

                EmbeddingEngine._shared_model_name = (
                    self.model_name
                )

                EmbeddingEngine._shared_device = (
                    self.device
                )

                EmbeddingEngine._shared_torch_device = (
                    run_device
                )

                EmbeddingEngine._shared_load_count += 1

                logger.info(
                    "Speaker embedding model loaded "
                    "successfully on %s.",
                    run_device,
                )

                logger.info(
                    "Shared embedding model ready "
                    "for all sessions and speakers."
                )

            except EmbeddingError as exc:

                EmbeddingEngine._shared_model = None

                EmbeddingEngine._shared_loaded = False

                EmbeddingEngine._shared_load_error = (
                    str(exc)
                )

                self._increment_error_metric()

                raise

            except Exception as exc:

                EmbeddingEngine._shared_model = None

                EmbeddingEngine._shared_loaded = False

                EmbeddingEngine._shared_load_error = (
                    str(exc)
                )

                self._increment_error_metric()

                logger.exception(
                    "Failed to load shared speaker "
                    "embedding model."
                )

                raise EmbeddingModelError(
                    "Unable to load speaker embedding "
                    f"model '{self.model_name}'."
                ) from exc

    # ================================================================
    # ENSURE MODEL
    # ================================================================

    def _ensure_loaded(
        self,
    ) -> None:

        if not self.enabled:

            raise EmbeddingError(
                "EmbeddingEngine is disabled."
            )

        if not EmbeddingEngine._shared_loaded:

            self.load()

        if (
            not EmbeddingEngine._shared_loaded
            or EmbeddingEngine._shared_model is None
        ):

            raise EmbeddingModelError(
                "Shared embedding model could not "
                "be loaded."
            )

    # ================================================================
    # AUDIO CONVERSION
    # ================================================================

    def _pcm16_to_numpy(
        self,
        audio: bytes | bytearray | memoryview,
    ) -> Any:

        if np is None:

            raise EmbeddingError(
                "NumPy is not installed."
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
                "Audio must be bytes, "
                "bytearray or memoryview."
            )

        raw = bytes(audio)

        if not raw:

            raise AudioFormatError(
                "Audio buffer is empty."
            )

        if (
            len(raw) % SAMPLE_WIDTH
            != 0
        ):

            raise AudioFormatError(
                "PCM16 audio byte length must be "
                "aligned to the sample width."
            )

        samples = np.frombuffer(
            raw,
            dtype="<i2",
        )

        if samples.size == 0:

            raise AudioFormatError(
                "No PCM16 samples found."
            )

        waveform = (
            samples.astype(
                np.float32
            )
            / 32768.0
        )

        if not np.all(
            np.isfinite(waveform)
        ):

            raise AudioFormatError(
                "PCM16 audio contains invalid values."
            )

        return waveform

    # ================================================================
    # AUDIO DURATION
    # ================================================================

    def _duration_ms(
        self,
        waveform: Any,
    ) -> float:

        if waveform is None:
            return 0.0

        return (
            len(waveform)
            / float(self.sample_rate)
            * 1000.0
        )

    def _validate_audio_duration(
        self,
        waveform: Any,
    ) -> None:

        duration_ms = (
            self._duration_ms(
                waveform
            )
        )

        if duration_ms < self.min_audio_ms:

            raise EmbeddingError(
                "Audio is too short for speaker "
                "embedding: "
                f"{duration_ms:.1f} ms < "
                f"{self.min_audio_ms} ms."
            )

        if duration_ms > self.max_audio_ms:

            raise EmbeddingError(
                "Audio is too long for speaker "
                "embedding: "
                f"{duration_ms:.1f} ms > "
                f"{self.max_audio_ms} ms."
            )

    # ================================================================
    # NORMALIZATION
    # ================================================================

    @staticmethod
    def _normalize_embedding(
        embedding: Any,
    ) -> Any:

        if np is None:

            raise EmbeddingError(
                "NumPy is required."
            )

        vector = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        if vector.size == 0:

            raise EmbeddingError(
                "Embedding vector is empty."
            )

        if not np.all(
            np.isfinite(vector)
        ):

            raise EmbeddingError(
                "Embedding contains invalid "
                "NaN or infinity values."
            )

        norm = float(
            np.linalg.norm(vector)
        )

        if (
            not math.isfinite(norm)
            or norm <= 1e-12
        ):

            raise EmbeddingError(
                "Embedding norm is invalid or zero."
            )

        normalized = (
            vector / norm
        ).astype(
            np.float32
        )

        return normalized

    # ================================================================
    # OUTPUT EXTRACTION
    # ================================================================

    def _extract_embedding(
        self,
        output: Any,
    ) -> Any:

        if output is None:

            raise EmbeddingModelError(
                "Embedding model returned None."
            )

        if (
            torch is not None
            and isinstance(
                output,
                torch.Tensor,
            )
        ):

            tensor = (
                output
                .detach()
                .float()
                .cpu()
            )

            if tensor.numel() == 0:

                raise EmbeddingModelError(
                    "Embedding model returned "
                    "an empty tensor."
                )

            # SpeechBrain ECAPA typically returns:
            #
            # [batch, 1, embedding_dimension]
            #
            # or:
            #
            # [batch, embedding_dimension]

            if tensor.ndim >= 3:

                tensor = tensor.reshape(
                    tensor.shape[0],
                    -1,
                )

            if (
                tensor.ndim >= 2
                and tensor.shape[0] == 1
            ):

                tensor = tensor[0]

            vector = (
                tensor
                .reshape(-1)
                .numpy()
            )

        else:

            if np is None:

                raise EmbeddingError(
                    "NumPy is required."
                )

            try:

                array = np.asarray(
                    output,
                    dtype=np.float32,
                )

                if array.size == 0:

                    raise EmbeddingModelError(
                        "Embedding model returned "
                        "an empty vector."
                    )

                if array.ndim >= 3:

                    array = array.reshape(
                        array.shape[0],
                        -1,
                    )

                if (
                    array.ndim >= 2
                    and array.shape[0] == 1
                ):

                    array = array[0]

                vector = array.reshape(-1)

            except EmbeddingModelError:
                raise

            except Exception as exc:

                raise EmbeddingModelError(
                    "Unable to convert embedding "
                    "model output to NumPy."
                ) from exc

        if vector.size == 0:

            raise EmbeddingModelError(
                "Embedding model returned "
                "an empty vector."
            )

        if not np.all(
            np.isfinite(vector)
        ):

            raise EmbeddingModelError(
                "Embedding model returned "
                "NaN or infinity."
            )

        return vector.astype(
            np.float32
        )

    # ================================================================
    # EMBEDDING
    # ================================================================

    def embed(
        self,
        audio: bytes | bytearray | memoryview,
        *,
        sample_rate: int = SAMPLE_RATE,
    ) -> tuple[float, ...]:
        """
        Génère un embedding ECAPA normalisé.

        Entrée :

            PCM16
            16000 Hz
            Mono

        Sortie :

            tuple[float, ...]
        """

        started = time.perf_counter()

        try:

            # --------------------------------------------------------
            # SAMPLE RATE
            # --------------------------------------------------------

            try:

                input_sample_rate = int(
                    sample_rate
                )

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise AudioFormatError(
                    "Invalid audio sample rate."
                ) from exc

            if (
                input_sample_rate
                != self.sample_rate
            ):

                raise AudioFormatError(
                    "Invalid embedding sample rate: "
                    f"{input_sample_rate} Hz. "
                    f"Expected {self.sample_rate} Hz."
                )

            # --------------------------------------------------------
            # PCM16 -> NUMPY
            # --------------------------------------------------------

            waveform = (
                self._pcm16_to_numpy(
                    audio
                )
            )

            # --------------------------------------------------------
            # DURATION
            # --------------------------------------------------------

            self._validate_audio_duration(
                waveform
            )

            # --------------------------------------------------------
            # MODEL
            # --------------------------------------------------------

            self._ensure_loaded()

            model = (
                EmbeddingEngine._shared_model
            )

            if model is None:

                raise EmbeddingModelError(
                    "Shared embedding model "
                    "is not initialized."
                )

            # --------------------------------------------------------
            # TENSOR
            # --------------------------------------------------------
            #
            # SpeechBrain EncoderClassifier
            # expects [batch, samples].
            #
            # Input remains on CPU.
            # SpeechBrain manages the model/device.
            #
            # One call represents one speaker segment.
            #
            # --------------------------------------------------------

            tensor = torch.from_numpy(
                waveform
            ).float()

            if tensor.ndim != 1:

                raise EmbeddingModelError(
                    "Invalid waveform shape: "
                    f"{tuple(tensor.shape)}."
                )

            tensor = tensor.unsqueeze(
                0
            )

            if tensor.ndim != 2:

                raise EmbeddingModelError(
                    "Invalid embedding input shape: "
                    f"{tuple(tensor.shape)}. "
                    "Expected [batch, samples]."
                )

            if tensor.shape[0] != 1:

                raise EmbeddingModelError(
                    "Embedding input must contain "
                    "exactly one audio sample."
                )

            # --------------------------------------------------------
            # INFERENCE
            # --------------------------------------------------------

            with EmbeddingEngine._shared_lock:

                with torch.inference_mode():

                    try:

                        output = (
                            model.encode_batch(
                                tensor,
                                normalize=False,
                            )
                        )

                    except TypeError as exc:

                        # ------------------------------------------------
                        # Compatibility with older SpeechBrain versions.
                        #
                        # Only fallback when the error concerns the
                        # normalize argument. Other TypeError exceptions
                        # must propagate normally.
                        # ------------------------------------------------

                        error_message = str(
                            exc
                        ).lower()

                        if "normalize" not in error_message:

                            raise

                        output = (
                            model.encode_batch(
                                tensor
                            )
                        )

            # --------------------------------------------------------
            # OUTPUT
            # --------------------------------------------------------

            vector = (
                self._extract_embedding(
                    output
                )
            )

            # --------------------------------------------------------
            # NORMALIZATION
            # --------------------------------------------------------

            vector = (
                self._normalize_embedding(
                    vector
                )
            )

            # --------------------------------------------------------
            # DIMENSION
            # --------------------------------------------------------

            dimension = int(
                vector.shape[0]
            )

            if (
                dimension
                != self.embedding_dimension
            ):

                raise EmbeddingDimensionError(
                    "Embedding dimension mismatch: "
                    f"model returned {dimension}, "
                    f"expected "
                    f"{self.embedding_dimension}."
                )

            # --------------------------------------------------------
            # PROCESSING TIME
            # --------------------------------------------------------

            processing_time = (
                time.perf_counter()
                - started
            )

            with self._lock:

                self._inference_count += 1

                self._total_processing_time += (
                    processing_time
                )

            try:

                metrics.add_processing_time(
                    processing_time
                )

            except Exception:

                logger.debug(
                    "Unable to update "
                    "embedding processing metrics.",
                    exc_info=True,
                )

            # --------------------------------------------------------
            # RESULT
            # --------------------------------------------------------

            return tuple(
                float(value)
                for value in vector.tolist()
            )

        except (
            AudioFormatError,
            EmbeddingError,
        ):

            self._increment_error_metric()

            logger.debug(
                "Embedding generation rejected.",
                exc_info=True,
            )

            raise

        except Exception as exc:

            self._increment_error_metric()

            logger.exception(
                "Speaker embedding generation failed."
            )

            raise EmbeddingError(
                "Speaker embedding generation failed."
            ) from exc

    # ================================================================
    # EMBEDDING RESULT
    # ================================================================

    def embed_result(
        self,
        audio: bytes | bytearray | memoryview,
        *,
        sample_rate: int = SAMPLE_RATE,
    ) -> EmbeddingResult:
        """
        Génère un embedding avec métadonnées.
        """

        started = time.perf_counter()

        if not isinstance(
            audio,
            (
                bytes,
                bytearray,
                memoryview,
            ),
        ):

            raise AudioFormatError(
                "Audio must be bytes, "
                "bytearray or memoryview."
            )

        try:

            input_sample_rate = int(
                sample_rate
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise AudioFormatError(
                "Invalid audio sample rate."
            ) from exc

        if (
            input_sample_rate
            != self.sample_rate
        ):

            raise AudioFormatError(
                "Invalid embedding sample rate: "
                f"{input_sample_rate} Hz. "
                f"Expected {self.sample_rate} Hz."
            )

        raw_audio = bytes(
            audio
        )

        if not raw_audio:

            raise AudioFormatError(
                "Audio buffer is empty."
            )

        if (
            len(raw_audio) % SAMPLE_WIDTH
            != 0
        ):

            raise AudioFormatError(
                "PCM16 audio byte length must be "
                "aligned to the sample width."
            )

        embedding = self.embed(
            raw_audio,
            sample_rate=input_sample_rate,
        )

        processing_time = (
            time.perf_counter()
            - started
        )

        duration_seconds = (
            len(raw_audio)
            / (
                float(input_sample_rate)
                * SAMPLE_WIDTH
                * CHANNELS
            )
        )

        return EmbeddingResult(
            embedding=embedding,
            dimension=len(
                embedding
            ),
            sample_rate=input_sample_rate,
            duration_seconds=duration_seconds,
            device=self.device,
            model=self.model_name,
            processing_time=processing_time,
            normalized=True,
        )

    # ================================================================
    # VECTOR
    # ================================================================

    @staticmethod
    def _vector(
        embedding: Iterable[float],
    ) -> Any:

        if np is None:

            raise EmbeddingError(
                "NumPy is required."
            )

        if embedding is None:

            raise EmbeddingError(
                "Embedding cannot be None."
            )

        try:

            vector = np.asarray(
                list(embedding),
                dtype=np.float32,
            ).reshape(-1)

        except Exception as exc:

            raise EmbeddingError(
                "Invalid embedding vector."
            ) from exc

        if vector.size == 0:

            raise EmbeddingError(
                "Embedding vector is empty."
            )

        if not np.all(
            np.isfinite(vector)
        ):

            raise EmbeddingError(
                "Embedding contains NaN "
                "or infinity."
            )

        return vector

    # ================================================================
    # COSINE SIMILARITY
    # ================================================================

    @classmethod
    def cosine_similarity(
        cls,
        embedding_a: Iterable[float],
        embedding_b: Iterable[float],
    ) -> float:

        vector_a = cls._vector(
            embedding_a
        )

        vector_b = cls._vector(
            embedding_b
        )

        if vector_a.shape != vector_b.shape:

            raise EmbeddingDimensionError(
                "Embedding dimensions do not match: "
                f"{vector_a.shape[0]} != "
                f"{vector_b.shape[0]}."
            )

        norm_a = float(
            np.linalg.norm(
                vector_a
            )
        )

        norm_b = float(
            np.linalg.norm(
                vector_b
            )
        )

        if (
            not math.isfinite(norm_a)
            or norm_a <= 1e-12
        ):

            raise EmbeddingError(
                "First embedding has an "
                "invalid or zero norm."
            )

        if (
            not math.isfinite(norm_b)
            or norm_b <= 1e-12
        ):

            raise EmbeddingError(
                "Second embedding has an "
                "invalid or zero norm."
            )

        score = float(
            np.dot(
                vector_a,
                vector_b,
            )
            / (
                norm_a
                * norm_b
            )
        )

        if not math.isfinite(score):

            raise EmbeddingError(
                "Cosine similarity is invalid."
            )

        return max(
            -1.0,
            min(
                1.0,
                score,
            ),
        )

    # ================================================================
    # COSINE DISTANCE
    # ================================================================

    @classmethod
    def cosine_distance(
        cls,
        embedding_a: Iterable[float],
        embedding_b: Iterable[float],
    ) -> float:

        similarity = (
            cls.cosine_similarity(
                embedding_a,
                embedding_b,
            )
        )

        distance = (
            1.0 - similarity
        )

        return max(
            0.0,
            min(
                2.0,
                distance,
            ),
        )

    # ================================================================
    # COMPARE
    # ================================================================

    @classmethod
    def compare(
        cls,
        embedding_a: Iterable[float],
        embedding_b: Iterable[float],
    ) -> dict[str, float]:

        similarity = (
            cls.cosine_similarity(
                embedding_a,
                embedding_b,
            )
        )

        distance = max(
            0.0,
            min(
                2.0,
                1.0 - similarity,
            ),
        )

        return {
            "similarity": round(
                similarity,
                6,
            ),
            "distance": round(
                distance,
                6,
            ),
        }

    # ================================================================
    # STATUS
    # ================================================================

    @property
    def is_loaded(self) -> bool:

        with EmbeddingEngine._shared_lock:

            return bool(
                EmbeddingEngine._shared_loaded
                and EmbeddingEngine._shared_model
                is not None
            )

    @property
    def is_available(self) -> bool:

        return bool(
            self.enabled
            and np is not None
            and torch is not None
            and EncoderClassifier is not None
        )

    # ================================================================
    # STATISTICS
    # ================================================================

    def statistics(
        self,
    ) -> dict[str, Any]:

        with self._lock:

            inference_count = (
                self._inference_count
            )

            total_processing_time = (
                self._total_processing_time
            )

        with EmbeddingEngine._shared_lock:

            load_error = (
                EmbeddingEngine._shared_load_error
            )

            shared_load_count = (
                EmbeddingEngine._shared_load_count
            )

            shared_model_name = (
                EmbeddingEngine._shared_model_name
            )

            shared_device = (
                EmbeddingEngine._shared_device
            )

            shared_torch_device = (
                EmbeddingEngine._shared_torch_device
            )

        average_time = (
            total_processing_time
            / inference_count
            if inference_count > 0
            else 0.0
        )

        return {
            "enabled": self.enabled,

            "loaded": self.is_loaded,

            "available": self.is_available,

            "model": self.model_name,

            "shared_model_name": (
                shared_model_name
            ),

            "device": self.device,

            "shared_device": (
                shared_device
            ),

            "torch_device": self.torch_device,

            "shared_torch_device": (
                shared_torch_device
            ),

            "sample_rate": self.sample_rate,

            "embedding_dimension": (
                self.embedding_dimension
            ),

            "inference_batch_size": (
                self.inference_batch_size
            ),

            "cpu_threads": (
                self.cpu_threads
            ),

            "gpu_device_index": (
                self.gpu_device_index
            ),

            "inference_count": (
                inference_count
            ),

            "total_processing_time": round(
                total_processing_time,
                6,
            ),

            "average_processing_time": round(
                average_time,
                6,
            ),

            "shared_model": True,

            "shared_load_count": (
                shared_load_count
            ),

            "load_error": load_error,

            "uptime_seconds": round(
                time.time()
                - self._created_at,
                3,
            ),
        }

    # ================================================================
    # HEALTH
    # ================================================================

    def health(
        self,
    ) -> dict[str, Any]:

        if not self.enabled:

            return {
                "status": "disabled",
                "component": "embedding_engine",
                "model": self.model_name,
                "device": self.device,
            }

        dependencies = {
            "numpy": np is not None,

            "torch": torch is not None,

            "speechbrain": (
                EncoderClassifier is not None
            ),
        }

        if not all(
            dependencies.values()
        ):

            return {
                "status": "unhealthy",
                "component": "embedding_engine",
                "reason": (
                    "Missing embedding dependencies."
                ),
                "dependencies": dependencies,
                "device": self.device,
            }

        if self.device == "cuda":

            cuda_available = (
                torch.cuda.is_available()
            )

            if not cuda_available:

                return {
                    "status": "unhealthy",
                    "component": "embedding_engine",
                    "model": self.model_name,
                    "device": self.device,
                    "torch_device": self.torch_device,
                    "loaded": self.is_loaded,
                    "reason": (
                        "CUDA is unavailable."
                    ),
                    "dependencies": dependencies,
                }

            if (
                self.gpu_device_index
                >= torch.cuda.device_count()
            ):

                return {
                    "status": "unhealthy",
                    "component": "embedding_engine",
                    "model": self.model_name,
                    "device": self.device,
                    "torch_device": self.torch_device,
                    "loaded": self.is_loaded,
                    "reason": (
                        "Configured GPU device "
                        "index is unavailable."
                    ),
                    "dependencies": dependencies,
                }

        with EmbeddingEngine._shared_lock:

            load_error = (
                EmbeddingEngine._shared_load_error
            )

            shared_model_name = (
                EmbeddingEngine._shared_model_name
            )

            shared_torch_device = (
                EmbeddingEngine._shared_torch_device
            )

        if load_error:

            return {
                "status": "unhealthy",
                "component": "embedding_engine",
                "model": self.model_name,
                "device": self.device,
                "loaded": self.is_loaded,
                "error": load_error,
                "dependencies": dependencies,
            }

        # ------------------------------------------------------------
        # SHARED MODEL CONSISTENCY
        # ------------------------------------------------------------

        if self.is_loaded:

            if (
                shared_model_name
                != self.model_name
            ):

                return {
                    "status": "unhealthy",
                    "component": "embedding_engine",
                    "model": self.model_name,
                    "device": self.device,
                    "loaded": True,
                    "reason": (
                        "Loaded shared model does not "
                        "match requested model."
                    ),
                    "shared_model_name": (
                        shared_model_name
                    ),
                    "dependencies": dependencies,
                }

            if (
                shared_torch_device
                != self.torch_device
            ):

                return {
                    "status": "unhealthy",
                    "component": "embedding_engine",
                    "model": self.model_name,
                    "device": self.device,
                    "loaded": True,
                    "reason": (
                        "Loaded shared model device "
                        "does not match requested device."
                    ),
                    "shared_torch_device": (
                        shared_torch_device
                    ),
                    "torch_device": (
                        self.torch_device
                    ),
                    "dependencies": dependencies,
                }

        return {
            "status": "healthy",

            "component": (
                "embedding_engine"
            ),

            "model": self.model_name,

            "device": self.device,

            "torch_device": (
                self.torch_device
            ),

            "loaded": self.is_loaded,

            "available": self.is_available,

            "shared_model": True,

            "shared_load_count": (
                EmbeddingEngine._shared_load_count
            ),

            "dependencies": dependencies,
        }

    # ================================================================
    # WARMUP
    # ================================================================

    def warmup(
        self,
        audio: Optional[
            bytes | bytearray | memoryview
        ] = None,
    ) -> None:

        self.load()

        if audio is not None:

            self.embed(
                audio
            )

    # ================================================================
    # UNLOAD
    # ================================================================

    def unload(
        self,
    ) -> None:
        """
        Ne décharge pas le modèle partagé.

        Conservé pour compatibilité avec l'architecture SpeakerIdentity.
        """

        logger.debug(
            "EmbeddingEngine.unload() ignored: "
            "shared model remains available for "
            "all sessions and speakers."
        )

    # ================================================================
    # SHARED MODEL STATUS
    # ================================================================

    @classmethod
    def shared_model_loaded(
        cls,
    ) -> bool:

        with cls._shared_lock:

            return bool(
                cls._shared_loaded
                and cls._shared_model is not None
            )

    @classmethod
    def shared_model_load_count(
        cls,
    ) -> int:

        with cls._shared_lock:

            return cls._shared_load_count

    # ================================================================
    # ERROR METRICS
    # ================================================================

    def _increment_error_metric(
        self,
    ) -> None:

        try:

            metrics.increment(
                "errors"
            )

        except Exception:

            logger.debug(
                "Unable to update error metrics.",
                exc_info=True,
            )

    # ================================================================
    # REPRESENTATION
    # ================================================================

    def __repr__(
        self,
    ) -> str:

        return (
            "<EmbeddingEngine "
            f"model='{self.model_name}', "
            f"device='{self.device}', "
            f"loaded={self.is_loaded}, "
            "shared=True, "
            f"dimension={self.embedding_dimension}>"
        )


# ====================================================================
# FACTORY
# ====================================================================

def create_embedding_engine(
    *,
    config: Optional[Any] = None,
    auto_load: bool = False,
) -> EmbeddingEngine:

    return EmbeddingEngine(
        config=config,
        auto_load=auto_load,
    )


# ====================================================================
# PUBLIC EXPORTS
# ====================================================================

__all__ = [
    "EmbeddingResult",
    "EmbeddingEngine",
    "create_embedding_engine",
]