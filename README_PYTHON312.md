# SpeakerIdentity — installation et test (Python 3.12)

## 1. Prérequis
- Python 3.12 (64 bits)
- Connexion Internet au premier lancement pour télécharger le modèle vocal SpeechBrain.
- Un navigateur récent (Chrome ou Edge) et un microphone.

## 2. Créer un environnement virtuel (Windows)
Depuis le dossier `SpeakerIdentity-IA-RealtimeDiarization` :

```bat
py -3.12 -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 3. Démarrer le serveur
```bat
python real_diarization_test\\server.py
```

Puis ouvrir : http://127.0.0.1:8010/

Autoriser l'accès au microphone dans le navigateur et cliquer sur **Start Microphone**.

## 4. Vérifier l'état du serveur
- http://127.0.0.1:8010/health
- http://127.0.0.1:8010/statistics

## 5. Lancer les tests automatisés
À la racine du projet :
```bat
python -m pytest diarization\\tests -q
```

Le test réel avec microphone nécessite un périphérique audio et un navigateur. Le premier chargement de SpeechBrain peut être long, car le modèle ECAPA est téléchargé.

## Remarque PyTorch
Les versions `torch` et `torchaudio` sont alignées. Pour une installation GPU/CUDA, remplacez-les par les versions correspondant à votre version CUDA depuis l'index officiel PyTorch. Le fichier fourni vise une installation standard CPU.
