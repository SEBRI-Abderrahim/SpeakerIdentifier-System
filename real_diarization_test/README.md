====================================================================
SpeakerIdentity IA - REAL DIARIZATION TEST
====================================================================

OBJECTIF
--------

Tester réellement :

    PC Microphone
          |
          v
       Browser
          |
          v
      WebSocket
          |
          v
       PCM16
      16000 Hz
       Mono
          |
          v
    SpeakerIdentity Pipeline
          |
          +---- AudioProcessor
          |
          +---- VAD
          |
          +---- SegmentManager
          |
          +---- EmbeddingEngine
          |
          +---- SpeakerManager
          |
          +---- SessionManager
          |
          v
      WebClient


====================================================================
1. INSTALLATION
====================================================================

Depuis :

C:\Users\a.sebri\Desktop\test

installer si nécessaire :

    pip install fastapi uvicorn


====================================================================
2. LANCEMENT
====================================================================

Double-cliquer :

    real_diarization_test\run_test.bat


Ou :

    python real_diarization_test\server.py


====================================================================
3. NAVIGATEUR
====================================================================

Ouvrir :

    http://127.0.0.1:8010/


====================================================================
4. TEST MICROPHONE
====================================================================

Cliquer :

    Start Microphone


Autoriser le microphone dans Chrome.


====================================================================
5. TEST PAROLE
====================================================================

Dire :

    Bonjour, ceci est un test SpeakerIdentity.


Le navigateur transmet :

    PCM16
    16000 Hz
    Mono


====================================================================
6. TEST MULTI-SPEAKER
====================================================================

Personne 1 :

    Bonjour, je suis le premier interlocuteur.


Personne 2 :

    Bonjour, je suis le deuxième interlocuteur.


Personne 1 :

    Je reprends la parole.


Résultat attendu :

    speaker_1
    speaker_2
    speaker_1


====================================================================
7. HEALTH
====================================================================

Navigateur :

    http://127.0.0.1:8010/health


====================================================================
8. STATISTICS
====================================================================

Navigateur :

    http://127.0.0.1:8010/statistics


====================================================================
FORMAT AUDIO
====================================================================

Codec       : PCM16
Sample rate : 16000 Hz
Channels    : 1
Width       : 2 bytes/sample


====================================================================