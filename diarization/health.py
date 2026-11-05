"""
====================================================================
SpeakerIdentity IA - Diarization Health
====================================================================

Projet       : SpeakerIdentity IA
Module       : Diarization
Fichier      : diarization/health.py

Auteur       : SEBRI Abderrahim
Organisation : SAM TEHNOLOGY Lab
Version      : 2.1.1
Date         : 14/08/2026

Description
------------

Gestion centralisée de l'état de santé du module de diarisation
SpeakerIdentity IA.

Responsabilités :

    - vérifier l'état des composants ;
    - vérifier la disponibilité du moteur ;
    - exposer un état global ;
    - permettre l'intégration avec l'API REST ;
    - permettre l'intégration avec le monitoring ;
    - supporter plusieurs composants simultanément ;
    - ne jamais interrompre le service en cas d'erreur d'un
      composant de health check.

États supportés :

    healthy
    degraded
    unhealthy

Architecture :

    API / Monitoring
           |
           v
    DiarizationHealth
           |
      +----+----+----------------+
      |         |                |
      v         v                v
     VAD     Embedding        Pipeline
               Engine
      |
      v
    Health Status

IMPORTANT
---------

Ce module ne réalise aucun traitement audio.

Il ne charge aucun modèle.

Il ne crée aucune ressource lourde.

Il peut donc être importé sans provoquer :

    - chargement GPU ;
    - chargement d'embeddings ;
    - initialisation WebSocket ;
    - création de sessions.

====================================================================
"""

from __future__ import annotations

# ====================================================================
# IMPORTS
# ====================================================================

import time

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ====================================================================
# CONSTANTES
# ====================================================================

HEALTHY = "healthy"
DEGRADED = "degraded"
UNHEALTHY = "unhealthy"


# ====================================================================
# HEALTH CHECK RESULT
# ====================================================================


@dataclass
class HealthCheckResult:
    """
    Résultat d'un health check de composant.
    """

    name: str

    status: str = HEALTHY

    message: Optional[str] = None

    latency_ms: float = 0.0

    details: Dict[str, Any] = field(
        default_factory=dict
    )

    checked_at: float = field(
        default_factory=time.time
    )

    # ----------------------------------------------------------------
    # POST INIT
    # ----------------------------------------------------------------

    def __post_init__(self) -> None:
        """
        Normalise et valide le résultat du health check.
        """

        self.name = str(
            self.name
        ).strip()

        if not self.name:
            raise ValueError(
                "Health check name cannot be empty."
            )

        self.status = str(
            self.status
        ).strip().lower()

        if self.status not in {
            HEALTHY,
            DEGRADED,
            UNHEALTHY,
        }:
            self.status = DEGRADED

        if self.message is not None:
            self.message = str(
                self.message
            )

        try:
            self.latency_ms = float(
                self.latency_ms
            )
        except (
            TypeError,
            ValueError,
        ):
            self.latency_ms = 0.0

        if self.latency_ms < 0:
            self.latency_ms = 0.0

        if self.details is None:
            self.details = {}

        elif not isinstance(
            self.details,
            dict,
        ):
            self.details = dict(
                self.details
            )

        try:
            self.checked_at = float(
                self.checked_at
            )
        except (
            TypeError,
            ValueError,
        ):
            self.checked_at = time.time()

    # ----------------------------------------------------------------
    # DICT
    # ----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit le résultat en dictionnaire.
        """

        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "latency_ms": round(
                self.latency_ms,
                3,
            ),
            "details": dict(
                self.details
            ),
            "checked_at": self.checked_at,
        }


# ====================================================================
# HEALTH CHECK REGISTRATION
# ====================================================================


@dataclass
class HealthCheck:
    """
    Définition d'un health check.

    ``component`` peut être :

        - un objet possédant health() ;
        - une fonction personnalisée ;
        - un callable retournant un dictionnaire ;
        - un callable retournant un booléen.
    """

    name: str

    component: Any

    enabled: bool = True

    critical: bool = False


# ====================================================================
# DIARIZATION HEALTH
# ====================================================================


class DiarizationHealth:
    """
    Gestionnaire global du health check SpeakerIdentity Diarization.

    Exemple :

        health = DiarizationHealth()

        health.register(
            "vad",
            vad,
            critical=True,
        )

        health.register(
            "embedding",
            embedding_engine,
        )

        result = health.check()
    """

    def __init__(
        self,
        components: Optional[
            Dict[str, Any]
        ] = None,
    ) -> None:

        self._checks: Dict[
            str,
            HealthCheck,
        ] = {}

        self.started_at = time.time()

        if components:

            for name, component in (
                components.items()
            ):

                self.register(
                    name=name,
                    component=component,
                )

    # =================================================================
    # REGISTER
    # =================================================================

    def register(
        self,
        name: str,
        component: Any,
        *,
        enabled: bool = True,
        critical: bool = False,
    ) -> None:
        """
        Enregistre un composant.

        Parameters
        ----------
        name:
            Nom unique du composant.

        component:
            Objet ou callable à tester.

        enabled:
            Active ou désactive le check.

        critical:
            Si True, une panne de ce composant rend le service
            globalement unhealthy.
        """

        name = str(
            name
        ).strip()

        if not name:
            raise ValueError(
                "Health component name cannot be empty."
            )

        if component is None:
            raise ValueError(
                f"Health component '{name}' cannot be None."
            )

        self._checks[name] = HealthCheck(
            name=name,
            component=component,
            enabled=bool(enabled),
            critical=bool(critical),
        )

    # =================================================================
    # UNREGISTER
    # =================================================================

    def unregister(
        self,
        name: str,
    ) -> bool:
        """
        Supprime un composant du health manager.

        Retourne True si le composant existait.
        """

        name = str(
            name
        ).strip()

        if name in self._checks:

            del self._checks[name]

            return True

        return False

    # =================================================================
    # ENABLE / DISABLE
    # =================================================================

    def enable(
        self,
        name: str,
    ) -> None:
        """
        Active un health check.
        """

        name = str(
            name
        ).strip()

        check = self._checks.get(
            name
        )

        if check is None:
            raise KeyError(
                f"Unknown health component: {name}"
            )

        check.enabled = True

    # -----------------------------------------------------------------

    def disable(
        self,
        name: str,
    ) -> None:
        """
        Désactive un health check.
        """

        name = str(
            name
        ).strip()

        check = self._checks.get(
            name
        )

        if check is None:
            raise KeyError(
                f"Unknown health component: {name}"
            )

        check.enabled = False

    # =================================================================
    # COMPONENT CHECK
    # =================================================================

    def _check_component(
        self,
        check: HealthCheck,
    ) -> HealthCheckResult:
        """
        Exécute le health check d'un composant.
        """

        started = time.perf_counter()

        try:

            component = check.component

            # ---------------------------------------------------------
            # Cas 1 : méthode health()
            # ---------------------------------------------------------

            health_method = getattr(
                component,
                "health",
                None,
            )

            if callable(
                health_method
            ):

                result = health_method()

            # ---------------------------------------------------------
            # Cas 2 : callable direct
            # ---------------------------------------------------------

            elif callable(component):

                result = component()

            # ---------------------------------------------------------
            # Cas 3 : composant sans health()
            # ---------------------------------------------------------

            else:

                result = {
                    "status": HEALTHY,
                    "message": (
                        "Component registered without "
                        "explicit health check."
                    ),
                }

            latency_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            return self._normalize_result(
                name=check.name,
                result=result,
                latency_ms=latency_ms,
            )

        except Exception as exc:

            latency_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            return HealthCheckResult(
                name=check.name,
                status=UNHEALTHY,
                message=str(exc)
                or type(exc).__name__,
                latency_ms=latency_ms,
                details={
                    "exception": type(
                        exc
                    ).__name__,
                },
            )

    # =================================================================
    # NORMALIZE RESULT
    # =================================================================

    def _normalize_result(
        self,
        name: str,
        result: Any,
        latency_ms: float,
    ) -> HealthCheckResult:
        """
        Normalise les différents formats retournés par les composants.
        """

        # -------------------------------------------------------------
        # HealthCheckResult
        # -------------------------------------------------------------

        if isinstance(
            result,
            HealthCheckResult,
        ):

            return HealthCheckResult(
                name=name,
                status=result.status,
                message=result.message,
                latency_ms=latency_ms,
                details=dict(
                    result.details
                ),
                checked_at=result.checked_at,
            )

        # -------------------------------------------------------------
        # Booléen
        # -------------------------------------------------------------

        if isinstance(
            result,
            bool,
        ):

            return HealthCheckResult(
                name=name,
                status=(
                    HEALTHY
                    if result
                    else UNHEALTHY
                ),
                latency_ms=latency_ms,
            )

        # -------------------------------------------------------------
        # Dictionnaire
        # -------------------------------------------------------------

        if isinstance(
            result,
            dict,
        ):

            status = str(
                result.get(
                    "status",
                    HEALTHY,
                )
            ).strip().lower()

            if status not in {
                HEALTHY,
                DEGRADED,
                UNHEALTHY,
            }:

                status = DEGRADED

            message = result.get(
                "message"
            )

            if message is not None:
                message = str(
                    message
                )

            details = dict(
                result
            )

            details.pop(
                "status",
                None,
            )

            details.pop(
                "message",
                None,
            )

            details.pop(
                "latency_ms",
                None,
            )

            details.pop(
                "checked_at",
                None,
            )

            details.pop(
                "name",
                None,
            )

            return HealthCheckResult(
                name=name,
                status=status,
                message=message,
                latency_ms=latency_ms,
                details=details,
            )

        # -------------------------------------------------------------
        # None
        # -------------------------------------------------------------

        if result is None:

            return HealthCheckResult(
                name=name,
                status=DEGRADED,
                message=(
                    "Health check returned no result."
                ),
                latency_ms=latency_ms,
            )

        # -------------------------------------------------------------
        # Valeur inconnue
        # -------------------------------------------------------------

        return HealthCheckResult(
            name=name,
            status=HEALTHY,
            message=(
                "Component responded successfully."
            ),
            latency_ms=latency_ms,
            details={
                "response_type": type(
                    result
                ).__name__,
            },
        )

    # =================================================================
    # CHECK
    # =================================================================

    def check(self) -> Dict[str, Any]:
        """
        Exécute tous les health checks.

        Retourne un résultat global compatible API/JSON.
        """

        started = time.perf_counter()

        components: Dict[
            str,
            Dict[str, Any],
        ] = {}

        overall_status = HEALTHY

        checked_count = 0

        healthy_count = 0

        degraded_count = 0

        unhealthy_count = 0

        critical_failure = False

        for name, health_check in (
            self._checks.items()
        ):

            # ---------------------------------------------------------
            # Composant désactivé
            # ---------------------------------------------------------

            if not health_check.enabled:

                components[name] = {
                    "name": name,
                    "status": "disabled",
                    "critical": (
                        health_check.critical
                    ),
                }

                continue

            # ---------------------------------------------------------
            # Exécution
            # ---------------------------------------------------------

            checked_count += 1

            result = self._check_component(
                health_check
            )

            components[name] = result.to_dict()

            # ---------------------------------------------------------
            # Statistiques
            # ---------------------------------------------------------

            if result.status == HEALTHY:

                healthy_count += 1

            elif result.status == DEGRADED:

                degraded_count += 1

                if overall_status == HEALTHY:

                    overall_status = DEGRADED

            else:

                unhealthy_count += 1

                if health_check.critical:

                    critical_failure = True

                overall_status = UNHEALTHY

        # -------------------------------------------------------------
        # Aucun composant enregistré/actif
        # -------------------------------------------------------------

        if checked_count == 0:

            overall_status = DEGRADED

        # -------------------------------------------------------------
        # Critical failure
        # -------------------------------------------------------------

        if critical_failure:

            overall_status = UNHEALTHY

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        return {
            "service": "speakeridentity_diarization",
            "status": overall_status,
            "timestamp": time.time(),
            "uptime_seconds": round(
                time.time()
                - self.started_at,
                3,
            ),
            "check_latency_ms": round(
                elapsed_ms,
                3,
            ),
            "components": components,
            "summary": {
                "registered": len(
                    self._checks
                ),
                "checked": checked_count,
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "critical_failure": (
                    critical_failure
                ),
            },
        }

    # =================================================================
    # IS HEALTHY
    # =================================================================

    def is_healthy(self) -> bool:
        """
        Retourne True uniquement si le service est healthy.
        """

        result = self.check()

        return (
            result["status"]
            == HEALTHY
        )

    # =================================================================
    # STATUS
    # =================================================================

    def status(self) -> str:
        """
        Retourne uniquement l'état global.
        """

        return str(
            self.check()[
                "status"
            ]
        )

    # =================================================================
    # HEALTH
    # =================================================================

    def health(self) -> Dict[str, Any]:
        """
        Alias standard utilisé par les autres composants SpeakerIdentity.
        """

        return self.check()

    # =================================================================
    # COMPONENTS
    # =================================================================

    @property
    def components(self) -> Dict[str, Any]:
        """
        Retourne les composants enregistrés.

        Une copie est retournée afin de ne pas exposer directement
        le dictionnaire interne.
        """

        return {
            name: check.component
            for name, check in (
                self._checks.items()
            )
        }

    # =================================================================
    # COUNT
    # =================================================================

    @property
    def component_count(self) -> int:
        """
        Nombre de composants enregistrés.
        """

        return len(
            self._checks
        )


# ====================================================================
# GLOBAL HEALTH INSTANCE
# ====================================================================

health = DiarizationHealth()


# ====================================================================
# PUBLIC API
# ====================================================================

__all__ = [
    # Status
    "HEALTHY",
    "DEGRADED",
    "UNHEALTHY",

    # Models
    "HealthCheckResult",
    "HealthCheck",

    # Manager
    "DiarizationHealth",

    # Global instance
    "health",
]