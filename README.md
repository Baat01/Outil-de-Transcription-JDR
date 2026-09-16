# 🎲 Retranscription Vocale JDR

Outil CLI Python **100% local** pour transcrire, diariser et résumer des sessions de jeu de rôle sur table.

## Fonctionnalités

| Étape | Description |
|-------|-------------|
| 📚 **Lore multi-PDF** | Extrait les entités de plusieurs PDFs, sauvegardées dans `contexts/<thème>.json` |
| 🎤 **Audio multi-fichiers** | Accepte un fichier unique **ou un dossier** de morceaux (tri naturel + concaténation via pydub) |
| 🗣️ **Diarisation** | Sépare les locuteurs avec Pyannote |
| 👥 **Mapping joueurs** | Remplace les labels `SPEAKER_XX` par les vrais noms (via `players.json`) |
| ✏️ **Correction** | Corrige l'orthographe des noms selon le lore via LLM Ollama |
| 📋 **Fiche MJ** | Génère une fiche structurée (PNJ, intrigues, objets…) |
| ⚡ **Résumé Joueurs** | Résumé épique 150 mots style Dragon Ball Z |

---

## Prérequis

- **Python 3.10 à 3.12** (recommandé pour la compatibilité PyTorch / WhisperX)
- [Ollama](https://ollama.com/) installé et lancé (`ollama serve`)
- Un GPU NVIDIA compatible CUDA (recommandé) ou CPU
- Un token HuggingFace ([obtenir ici](https://huggingface.co/settings/tokens)) pour la diarisation
- **[ffmpeg](https://ffmpeg.org/)** installé sur le système *(requis pour la concaténation multi-fichiers)*
  - Windows : `winget install ffmpeg`
  - Linux   : `sudo apt install ffmpeg`
  - macOS   : `brew install ffmpeg`

---

## ⚡ Installation Automatisée (Recommandée)

Un script DevOps se charge de tout configurer automatiquement (création du venv, PyTorch CUDA, dépendances, Ollama model, configuration du token) :

```powershell
# Windows PowerShell
PowerShell -ExecutionPolicy Bypass -File .\setup.ps1
```

*(Ou en Python : `python setup.py`)*

Consultez le fichier **[ACTIONS_REQUISES.md](file:///c:/Users/bruellan/Documents/Project/Training/Retranscription%20JDR/ACTIONS_REQUISES.md)** pour les 2 licences HuggingFace à accepter manuellement.

---

## 🛠️ Installation Manuelle (Optionnelle)

### 1. Cloner / se placer dans le projet

```bash
cd "Retranscription JDR"
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Installer PyTorch (AVANT whisperx)

**GPU CUDA 11.8 :**
```bash
pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

**CPU uniquement :**
```bash
pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 5. Télécharger un modèle Ollama

```bash
ollama pull mistral:7b-instruct
```

### 6. Configurer le token HuggingFace (pour la diarisation)

Acceptez la licence Pyannote :
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

Puis définissez la variable d'environnement :

```bash
# Windows PowerShell
$env:HF_TOKEN = "hf_votre_token_ici"

# Linux / macOS
export HF_TOKEN="hf_votre_token_ici"
```

---

## Utilisation

### Commande de base

```bash
python main.py --audio session.mp3 --pdf lore.pdf
```

### Options complètes

```
usage: retranscription-jdr [-h] --audio AUDIO_FILE --pdf PDF_FILE
                           [--model MODEL] [--output-dir DIR]
                           [--whisper-model MODEL] [--device {cuda,cpu}]
                           [--no-diarization] [--hf-token TOKEN]
                           [--chunk-size WORDS] [--verbose]

Arguments obligatoires :
  --audio, -a       Fichier audio (mp3, wav, m4a, flac…)
  --pdf,   -p       PDF de lore de la campagne

Arguments optionnels :
  --model, -m       Modèle Ollama (défaut: mistral:7b-instruct)
  --output-dir, -o  Dossier de sortie (défaut: output/)
  --whisper-model   Modèle Whisper (défaut: large-v2)
  --device          cuda ou cpu (défaut: cuda)
  --no-diarization  Désactive Pyannote (pas besoin de HF_TOKEN)
  --hf-token        Token HuggingFace (priorité sur HF_TOKEN)
  --chunk-size      Mots max par chunk LLM (défaut: 2000)
  --verbose, -v     Logs de débogage détaillés
```

### Exemples

```bash
# Session avec un autre modèle Ollama
python main.py --audio session.wav --pdf lore.pdf --model llama3.1:8b

# Sans diarisation (si pas de token HuggingFace)
python main.py --audio session.mp3 --pdf lore.pdf --no-diarization

# Sur CPU uniquement
python main.py --audio session.mp3 --pdf lore.pdf --device cpu --whisper-model medium

# Avec logs détaillés
python main.py --audio session.mp3 --pdf lore.pdf --verbose
```

---

## 🎧 Gestion des Sessions Multi-Fichiers

Si votre session est enregistrée en **plusieurs morceaux** (ex: une pause a coupé l'enregistrement), passez simplement le **dossier** contenant les fichiers :

```bash
# Structure d'exemple
session_04/
  ├── part_1.mp3
  ├── part_2.mp3
  └── part_10.mp3

# Lancement
python main.py transcribe --audio ./session_04/ --context "Naruto"
```

Le script :
1. **Liste** tous les fichiers audio valides (`.mp3`, `.wav`, `.m4a`, `.flac`…)
2. **Trie** par ordre naturel (`part_1`, `part_2`, `part_10` et non `part_1`, `part_10`, `part_2`)
3. **Concatène** en un seul fichier WAV temporaire via `pydub`
4. **Transcrit** ce fichier unique avec WhisperX
5. **Supprime** automatiquement le fichier temporaire en fin de traitement

> **⚠️ Prérequis `pydub`** : La concaténation audio nécessite que `ffmpeg` soit installé sur votre système :
> - Windows : `winget install ffmpeg`
> - Linux   : `sudo apt install ffmpeg`
> - macOS   : `brew install ffmpeg`

---

## Variables d'environnement

| Variable | Description | Défaut |
|----------|-------------|--------|
| `HF_TOKEN` | Token HuggingFace (Pyannote) | — |
| `JDR_OLLAMA_MODEL` | Modèle Ollama | `mistral:7b-instruct` |
| `JDR_WHISPER_MODEL` | Modèle Whisper | `large-v2` |
| `JDR_DEVICE` | Device PyTorch | `cuda` |
| `JDR_COMPUTE_TYPE` | Type de calcul Whisper | `float16` |
| `JDR_BATCH_SIZE` | Batch size WhisperX | `16` |
| `JDR_OUTPUT_DIR` | Dossier de sortie | `output` |
| `JDR_CONTEXTS_DIR` | Dossier des contextes JSON | `contexts` |

---

## Fichiers de sortie

Tous les fichiers sont générés dans le dossier `output/` (ou `--output-dir`) :

| Fichier | Description |
|---------|-------------|
| `transcript_session.txt` | Retranscription diarisée lisible `[HH:MM:SS] Joueur : "texte"` |
| `transcript_raw.txt` | Transcription brute horodatée (format interne) |
| `transcript_corrected.txt` | Transcription avec noms de lore corrigés |
| `fiche_mj.md` | Fiche récapitulative Markdown pour le MJ |
| `resume_joueurs.md` | Résumé épique pour les joueurs |

Les contextes lore sont sauvégardés dans `contexts/<theme>.json` (éditables manuellement).

---

## Architecture du projet

```
Retranscription JDR/
├── main.py                   # Point d'entrée CLI (sous-commandes)
├── config.py                 # Configuration globale
├── requirements.txt          # Dépendances
├── players.json              # Exemple de mapping joueurs
├── README.md
├── contexts/                 # Contextes lore JSON (auto-créé)
│   └── naruto.json
├── pdfs/                     # Vos PDFs de lore
├── output/                   # Fichiers générés (auto-créé)
└── modules/
    ├── __init__.py
    ├── ollama_client.py      # Wrapper Ollama + health check
    ├── lore.py               # Extraction multi-PDF + contexte JSON + merge LLM
    ├── audio.py              # WhisperX + Pyannote + concat multi-fichiers (pydub)
    ├── chunking.py           # Formatage + découpage en chunks
    └── summarizer.py         # Correction + résumés LLM
```

---

## Dépannage

### `RuntimeError: Impossible de joindre le serveur Ollama`
→ Lancez Ollama : `ollama serve`

### `RuntimeError: Le modèle 'xxx' n'est pas installé`
→ Installez-le : `ollama pull mistral:7b-instruct`

### `CouldntDecodeError` ou erreur pydub lors de la concaténation
→ ffmpeg n'est pas installé ou pas dans le PATH.
  - Windows : `winget install ffmpeg`  puis redémarrez votre terminal
  - Linux   : `sudo apt install ffmpeg`
  - macOS   : `brew install ffmpeg`

### `CUDA out of memory`
→ Réduisez `--chunk-size` ou utilisez `--whisper-model medium` et `JDR_BATCH_SIZE=8`

### Diarisation désactivée automatiquement
→ Définissez `HF_TOKEN` ou passez `--hf-token votre_token`

### Qualité de transcription insuffisante
→ Essayez `--whisper-model large-v3` ou vérifiez la qualité du fichier audio
