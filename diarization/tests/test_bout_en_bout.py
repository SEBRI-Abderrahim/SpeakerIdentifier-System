"""
====================================================================
SpeakerIdentity IA - Diarization
Test de bout en bout : PCM16 -> deux interlocuteurs distincts
====================================================================

Ce fichier est nouveau. Il vérifie la seule chose qui compte
vraiment : qu'en donnant au pipeline deux voix qui alternent, on
obtient bien DEUX identifiants de locuteur, attribués de façon
stable.

Le moteur d'empreinte vocale réel exige speechbrain et le
téléchargement du modèle ECAPA depuis HuggingFace. Ce test le
remplace par un moteur de substitution qui calcule une empreinte
spectrale simple : de vrais vecteurs, réellement différents d'une
voix à l'autre, mais sans dépendance externe.

Le but n'est donc PAS de mesurer la qualité du modèle, mais de
vérifier que toute la plomberie tient debout :

    PCM16 -> VAD -> segments -> empreintes -> locuteurs -> événements

C'est précisément cette chaîne qui était cassée en trois endroits :

    1. segment_manager n'émettait aucun segment sur parole continue
       (comparaison de flottants) ;
    2. le pipeline transmettait l'objet EmbeddingResult au lieu du
       vecteur, ce qui faisait échouer CHAQUE segment sur
       "Invalid speaker embedding" ;
    3. speaker_manager fusionnait deux locuteurs dès qu'un segment
       ambigu passait (max sur tout l'historique).

====================================================================
"""

import numpy as np

import pytest

from diarization.embedding_engine import EmbeddingResult
from diarization.pipeline import DiarizationPipeline


SAMPLE_RATE = 16000

DIMENSION = 32


# ====================================================================
# MOTEUR D'EMPREINTE DE SUBSTITUTION
# ====================================================================

class MoteurEmpreinteSpectrale:
    """
    Empreinte = énergie du segment répartie en 32 bandes de fréquence.

    Déterministe, sans dépendance, et suffisamment discriminant pour
    séparer deux timbres différents.
    """

    dimension = DIMENSION

    is_loaded = True

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def health(self) -> dict:
        return {"status": "healthy"}

    def statistics(self) -> dict:
        return {}

    def embed(self, audio: bytes, **_) -> EmbeddingResult:

        echantillons = np.frombuffer(
            audio,
            dtype="<i2",
        ).astype(np.float32)

        if echantillons.size < 256:
            echantillons = np.pad(
                echantillons,
                (0, 256 - echantillons.size),
            )

        spectre = np.abs(
            np.fft.rfft(
                echantillons * np.hanning(echantillons.size)
            )
        )

        bandes = np.array_split(
            spectre[: spectre.size // 2],
            DIMENSION,
        )

        vecteur = np.array(
            [b.mean() for b in bandes],
            dtype=np.float32,
        )

        vecteur = vecteur / (np.linalg.norm(vecteur) + 1e-9)

        return EmbeddingResult(
            embedding=tuple(float(x) for x in vecteur),
            dimension=DIMENSION,
            sample_rate=SAMPLE_RATE,
            duration_seconds=len(audio) / (SAMPLE_RATE * 2),
            device="cpu",
            model="empreinte-spectrale-de-test",
            processing_time=0.0,
            normalized=True,
        )


# ====================================================================
# SIGNAUX
# ====================================================================

def voix(fondamentale: float, secondes: float, amplitude: int = 8000) -> bytes:
    """Voix synthétique : fondamentale + deux harmoniques."""

    t = np.arange(int(SAMPLE_RATE * secondes)) / SAMPLE_RATE

    signal = (
        np.sin(2 * np.pi * fondamentale * t)
        + 0.5 * np.sin(2 * np.pi * 2 * fondamentale * t)
        + 0.25 * np.sin(2 * np.pi * 3 * fondamentale * t)
    )

    return (signal / 1.75 * amplitude).astype(np.int16).tobytes()


def silence(secondes: float = 0.6) -> bytes:

    return np.zeros(
        int(SAMPLE_RATE * secondes),
        dtype=np.int16,
    ).tobytes()


# ====================================================================
# TESTS
# ====================================================================

@pytest.fixture
def pipeline():

    p = DiarizationPipeline(
        embedding_engine=MoteurEmpreinteSpectrale(),
    )

    # Le seuil est FIXE ici, volontairement.
    #
    # Ce test verifie la plomberie, pas le reglage : il ne doit pas
    # casser quand on ajuste SPEAKER_SIMILARITY_THRESHOLD dans la
    # configuration. La valeur est choisie pour l'empreinte spectrale
    # de substitution utilisee ci-dessus, qui ne se comporte pas comme
    # ECAPA : deux tons synthetiques differents y arrivent vers 0,38,
    # la ou deux personnes differentes tombent plutot vers 0,10 avec le
    # vrai modele.
    #
    # Le reglage du seuil reel se fait avec calibrer_seuil.py, sur de
    # vraies voix.

    p.speaker_manager.config.speaker_similarity_threshold = 0.60

    p.start_session(session_id="bout_en_bout")

    yield p

    p.stop_session("bout_en_bout")


def _locuteurs_par_tour(pipeline, tours):
    """Fait parler chaque voix à son tour, renvoie les speaker_id vus."""

    observes = []

    for fondamentale in tours:

        pipeline.process(
            session_id="bout_en_bout",
            audio=voix(fondamentale, 1.2),
        )

        resultat = pipeline.process(
            session_id="bout_en_bout",
            audio=silence(),
        )

        for evenement in resultat.events or []:

            donnees = getattr(evenement, "data", None) or {}

            if "speaker_id" in donnees:

                observes.append(donnees["speaker_id"])

                break

    return observes


def test_deux_voix_donnent_deux_locuteurs(pipeline):

    # A, B, A, B, A, B
    tours = [120, 240, 120, 240, 120, 240]

    observes = _locuteurs_par_tour(pipeline, tours)

    assert len(observes) == len(tours), (
        "Un segment par prise de parole attendu, obtenu %d." % len(observes)
    )

    assert len(set(observes)) == 2, (
        "Deux locuteurs attendus, obtenu %d : %s"
        % (len(set(observes)), set(observes))
    )

    # L'attribution doit être stable : A toujours le même identifiant.
    assert observes[0] == observes[2] == observes[4]
    assert observes[1] == observes[3] == observes[5]

    assert observes[0] != observes[1]


def test_une_seule_voix_ne_cree_quun_locuteur(pipeline):

    observes = _locuteurs_par_tour(pipeline, [120, 120, 120, 120])

    assert len(observes) == 4

    assert len(set(observes)) == 1, (
        "Une seule voix ne doit produire qu'un locuteur, obtenu %s"
        % set(observes)
    )


def test_parole_continue_produit_des_segments(pipeline):
    """
    Régression du bug de comparaison de flottants.

    14 s de parole ininterrompue produisaient ZÉRO segment, et tout
    l'audio au-delà de 10 s était détruit sans message.
    """

    resultats = [
        pipeline.process(
            session_id="bout_en_bout",
            audio=voix(120, 2.0),
        )
        for _ in range(7)
    ]

    segments = sum(
        len(r.segments or [])
        for r in resultats
    )

    assert segments >= 1, (
        "Aucun segment émis sur 14 s de parole continue : "
        "le segment actif est de nouveau bloqué."
    )
