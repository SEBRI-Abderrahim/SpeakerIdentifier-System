"""
====================================================================
SpeakerIdentity IA - Diarization
Integration Test

WebClient / Microphone
        |
        v
    WebSocket
        |
        v
    Diarization
        |
        v
     Results
====================================================================
"""

import asyncio
import math
import struct

import pytest


try:
    import websockets
except ImportError:

    websockets = None


# CORRECTIF : ces trois tests exigent un serveur SpeakerIdentity en ecoute sur
# 127.0.0.1:8010. Quand il n'y en a pas, ils appelaient pytest.fail(),
# donc toute la suite passait au rouge sur une machine de developpement
# ou le serveur n'est simplement pas lance. Un test d'integration dont
# la dependance est absente doit etre IGNORE, pas mis en echec :
# sinon plus personne ne regarde le resultat de la suite.
#
# Ils sont donc ignores quand le serveur est injoignable, et executes
# normalement des qu'il tourne :
#
#     python real_diarization_test/server.py
#
# Un vrai echec de protocole reste bien un echec.


def _serveur_absent(exc: BaseException) -> bool:
    """Distingue 'serveur pas la' d'une vraie erreur de protocole."""

    if isinstance(
        exc,
        (
            ConnectionRefusedError,
            OSError,
            TimeoutError,
            asyncio.TimeoutError,
        ),
    ):
        return True

    texte = str(exc).lower()

    return any(
        motif in texte
        for motif in (
            "refused",
            "connect call failed",
            "timed out",
            "cannot connect",
            "no route to host",
        )
    )


WEBSOCKET_URL = (
    "ws://127.0.0.1:8010/ws/diarization"
)


SAMPLE_RATE = 16000


def generate_pcm16_audio(
    duration_seconds: float = 1.0,
    frequency: float = 440.0,
) -> bytes:

    samples = int(
        SAMPLE_RATE
        * duration_seconds
    )

    data = bytearray()

    for index in range(samples):

        value = int(
            12000
            * math.sin(
                2.0
                * math.pi
                * frequency
                * index
                / SAMPLE_RATE
            )
        )

        data.extend(
            struct.pack(
                "<h",
                value,
            )
        )

    return bytes(data)


@pytest.mark.asyncio
async def test_microphone_websocket_connection():

    if websockets is None:

        pytest.skip(
            "websockets package is not installed."
        )

    try:

        async with websockets.connect(
            WEBSOCKET_URL,
            open_timeout=5,
        ) as websocket:

            assert websocket is not None

    except Exception as exc:

        if _serveur_absent(exc):

            pytest.skip(
                "Serveur SpeakerIdentity absent sur "
                f"{WEBSOCKET_URL} : test d'integration ignore."
            )

        pytest.fail(
            "Unable to connect to SpeakerIdentity diarization "
            f"WebSocket at {WEBSOCKET_URL}: {exc}"
        )


@pytest.mark.asyncio
async def test_microphone_pcm16_websocket():

    if websockets is None:

        pytest.skip(
            "websockets package is not installed."
        )

    audio = generate_pcm16_audio(
        duration_seconds=1.0
    )

    assert len(audio) > 0

    try:

        async with websockets.connect(
            WEBSOCKET_URL,
            open_timeout=5,
        ) as websocket:

            await websocket.send(
                audio
            )

            try:

                result = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=10,
                )

            except asyncio.TimeoutError:

                pytest.fail(
                    "No diarization result received "
                    "from WebSocket."
                )

            assert result is not None

    except (ConnectionRefusedError, OSError):

        pytest.skip(
            "Serveur SpeakerIdentity absent : test d'integration ignore."
        )


@pytest.mark.asyncio
async def test_microphone_stream_multiple_chunks():

    if websockets is None:

        pytest.skip(
            "websockets package is not installed."
        )

    try:

        async with websockets.connect(
            WEBSOCKET_URL,
            open_timeout=5,
        ) as websocket:

            audio = generate_pcm16_audio(
                duration_seconds=2.0
            )

            chunk_size = (
                SAMPLE_RATE
                * 2
                // 10
            )

            for position in range(
                0,
                len(audio),
                chunk_size,
            ):

                chunk = audio[
                    position:
                    position + chunk_size
                ]

                if chunk:

                    await websocket.send(
                        chunk
                    )

                    await asyncio.sleep(
                        0.01
                    )

            try:

                result = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=10,
                )

            except asyncio.TimeoutError:

                pytest.fail(
                    "No diarization result received "
                    "after streaming audio chunks."
                )

            assert result is not None

    except (ConnectionRefusedError, OSError):

        pytest.skip(
            "Serveur SpeakerIdentity absent : test d'integration ignore."
        )