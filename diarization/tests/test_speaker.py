"""
====================================================================
SpeakerIdentity IA - Diarization
Tests Speaker  (moteur de diarisation)
====================================================================

CORRECTIF
---------

Ce fichier testait une API qui n'a jamais existe :

    await engine.load_model()
    await engine.create_session(...)
    await engine.attach_transcription(...)

DiarizationEngine est SYNCHRONE, sa methode de demarrage s'appelle
start(), et attach_transcription n'existe nulle part dans le projet
(verifie par recherche sur l'ensemble des sources).

Le test echouait donc systematiquement sur

    AttributeError: 'DiarizationEngine' object has no attribute
    'load_model'

Il teste desormais ce que le moteur fait reellement : demarrer, ouvrir
une session, avaler du PCM16 et rendre un resultat exploitable.

====================================================================
"""

import pytest

from diarization.diarization_engine import (
    DiarizationEngine,
)

from diarization.exceptions import (
    SessionNotFoundError,
)


SAMPLE_RATE = 16000


def _pcm16_silence(duree_ms: int) -> bytes:
    """Bloc PCM16 mono 16 kHz, silencieux."""

    echantillons = int(SAMPLE_RATE * duree_ms / 1000)

    return b"\x00\x00" * echantillons


@pytest.fixture
def moteur():

    engine = DiarizationEngine()

    engine.start()

    yield engine

    engine.stop()


def test_engine_demarre(moteur):

    assert moteur.running is True

    assert moteur.state in {
        "running",
        "started",
        "ready",
    }


def test_engine_cree_une_session(moteur):

    session = moteur.create_session("session_1")

    assert session is not None

    assert moteur.session_count() >= 1

    assert moteur.get_session("session_1") is not None


def test_engine_traite_du_pcm16(moteur):

    moteur.create_session("session_1")

    resultat = moteur.process_audio(
        session_id="session_1",
        audio=_pcm16_silence(100),
    )

    assert resultat is not None


def test_engine_health_et_statistiques(moteur):

    moteur.create_session("session_1")

    sante = moteur.health()

    assert isinstance(sante, dict)

    assert "status" in sante

    stats = moteur.statistics()

    assert isinstance(stats, dict)


def test_engine_sessions_isolees(moteur):

    moteur.create_session("session_a")
    moteur.create_session("session_b")

    assert moteur.session_count() >= 2

    assert moteur.get_session("session_a") is not None
    assert moteur.get_session("session_b") is not None

    moteur.delete_session("session_a")

    # get_session leve SessionNotFoundError sur une session absente,
    # il ne renvoie pas None.
    with pytest.raises(SessionNotFoundError):
        moteur.get_session("session_a")

    assert moteur.get_session("session_b") is not None
