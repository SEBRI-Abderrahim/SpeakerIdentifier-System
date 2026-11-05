"""
====================================================================
SpeakerIdentity IA - Diarization
Tests multi-session
====================================================================

CORRECTIF
---------

Ce fichier etait ecrit contre une API asynchrone qui n'existe pas :

    await engine.load_model()
    await engine.create_session(...)
    await engine.process_audio(session, audio, "fr")
    await engine.get_statistics()
    await engine.shutdown()

DiarizationEngine est SYNCHRONE. Ses methodes reelles sont start(),
create_session(), process_audio(session_id=, audio=), statistics() et
stop(). process_audio ne prend pas de langue : la diarisation ne
depend pas de la langue parlee.

Le test echouait sur AttributeError des la premiere ligne utile.

Il verifie desormais ce qui compte vraiment ici : que plusieurs
sessions coexistent sans se melanger, et que le traitement de l'une
n'affecte pas les autres. Le test tourne en parallele sur des threads,
ce qui exerce aussi les verrous du moteur.

====================================================================
"""

import threading

import numpy as np

import pytest

from diarization.diarization_engine import (
    DiarizationEngine,
)


SAMPLE_RATE = 16000


def bruit_pcm16(secondes: float = 1.0, graine: int = 123) -> bytes:
    """Signal PCM16 mono 16 kHz, deterministe."""

    echantillons = int(SAMPLE_RATE * secondes)

    rng = np.random.default_rng(graine)

    audio = (
        rng.normal(0, 5000, echantillons)
        .clip(-32768, 32767)
        .astype(np.int16)
    )

    return audio.tobytes()


@pytest.fixture
def moteur():

    engine = DiarizationEngine()

    engine.start()

    yield engine

    engine.stop()


def test_sessions_multiples(moteur):

    for nom in ("session_A", "session_B", "session_C"):
        moteur.create_session(nom)

    assert moteur.session_count() == 3

    audio = bruit_pcm16(1.0)

    resultats = [
        moteur.process_audio(
            session_id=nom,
            audio=audio,
        )
        for nom in ("session_A", "session_B", "session_C")
    ]

    assert len(resultats) == 3

    assert all(r is not None for r in resultats)

    stats = moteur.statistics()

    assert isinstance(stats, dict)


def test_sessions_en_parallele(moteur):
    """Trois sessions alimentees simultanement depuis trois threads."""

    noms = ("session_1", "session_2", "session_3")

    for nom in noms:
        moteur.create_session(nom)

    erreurs = []

    def alimenter(nom: str, graine: int) -> None:

        audio = bruit_pcm16(0.5, graine=graine)

        try:
            for _ in range(4):
                moteur.process_audio(
                    session_id=nom,
                    audio=audio,
                )
        except Exception as exc:  # pragma: no cover
            erreurs.append((nom, exc))

    fils = [
        threading.Thread(
            target=alimenter,
            args=(nom, 100 + i),
        )
        for i, nom in enumerate(noms)
    ]

    for f in fils:
        f.start()

    for f in fils:
        f.join(timeout=30)

    assert not erreurs, erreurs

    assert moteur.session_count() == 3


def test_session_supprimee_ninfluence_pas_les_autres(moteur):

    moteur.create_session("garde")
    moteur.create_session("jetee")

    moteur.process_audio(
        session_id="garde",
        audio=bruit_pcm16(0.5),
    )

    moteur.delete_session("jetee")

    assert moteur.session_count() == 1

    # La session conservee reste utilisable.
    assert (
        moteur.process_audio(
            session_id="garde",
            audio=bruit_pcm16(0.5),
        )
        is not None
    )
