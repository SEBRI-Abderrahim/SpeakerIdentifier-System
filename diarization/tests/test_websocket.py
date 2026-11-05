"""
====================================================================
SpeakerIdentity IA - Diarization
Tests WebSocket
====================================================================

CORRECTIF
---------

Ce fichier contenait, a l'interieur d'une fonction :

    from diarization.websocket import *

C'est une SyntaxError en Python 3 : "import * only allowed at module
level". Le fichier ne compilait donc pas, et pytest s'arretait a la
collecte AVANT d'executer le moindre test de la suite.

====================================================================
"""

import diarization.websocket as websocket

from diarization.websocket import (
    DiarizationWebSocket,
    WebSocketSession,
    create_websocket_handler,
)


def test_websocket_module():

    assert websocket is not None


def test_websocket_symboles():

    assert DiarizationWebSocket is not None

    assert WebSocketSession is not None

    assert create_websocket_handler is not None


def test_websocket_handler_creation():

    # create_websocket_handler exige un moteur : c'est lui qui traite
    # l'audio recu sur la socket.

    from diarization.diarization_engine import DiarizationEngine

    engine = DiarizationEngine()
    engine.start()

    handler = create_websocket_handler(engine)

    assert handler is not None

    assert isinstance(
        handler,
        DiarizationWebSocket,
    )

    engine.stop()
