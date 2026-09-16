"""
config.py - Configuration globale de l'outil de retranscription JDR.
Toutes les constantes ajustables sont centralisées ici.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Modèle Ollama utilisé par défaut (remplacez par tout modèle installé localement)
OLLAMA_MODEL: str = os.getenv("JDR_OLLAMA_MODEL", "mistral:7b-instruct")

# Timeout en secondes pour les appels Ollama
OLLAMA_TIMEOUT: int = 300

# ---------------------------------------------------------------------------
# WhisperX
# ---------------------------------------------------------------------------

# Modèle Whisper à utiliser : tiny, base, small, medium, large-v1, large-v2, large-v3
WHISPER_MODEL: str = os.getenv("JDR_WHISPER_MODEL", "large-v2")

# Langue de la session (laisser None pour détection automatique)
WHISPER_LANGUAGE: str | None = os.getenv("JDR_LANGUAGE", "fr")

# Device de calcul : "cuda" ou "cpu"
WHISPER_DEVICE: str = os.getenv("JDR_DEVICE", "cuda")

# Type de calcul : "float16" (GPU), "int8" (CPU/GPU économe), "float32"
WHISPER_COMPUTE_TYPE: str = os.getenv("JDR_COMPUTE_TYPE", "float16")

# Taille des batchs pour WhisperX (réduire si manque de VRAM)
WHISPER_BATCH_SIZE: int = int(os.getenv("JDR_BATCH_SIZE", "16"))

# ---------------------------------------------------------------------------
# Pyannote (Diarisation)
# ---------------------------------------------------------------------------

# Token HuggingFace requis pour le modèle Pyannote
# Obtenez-en un sur https://huggingface.co/settings/tokens
# et acceptez la licence sur https://huggingface.co/pyannote/speaker-diarization-3.1
HF_TOKEN: str | None = os.getenv("HF_TOKEN")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# Nombre maximum de mots par chunk envoyé au LLM
CHUNK_MAX_WORDS: int = 2000

# Chevauchement entre chunks (en mots) pour éviter les coupures de contexte
CHUNK_OVERLAP_WORDS: int = 100

# ---------------------------------------------------------------------------
# Sortie
# ---------------------------------------------------------------------------

# Dossier de sortie par défaut (sera créé si absent)
OUTPUT_DIR: str = os.getenv("JDR_OUTPUT_DIR", "output")

# Dossier des contextes lore thématiques (JSON éditables)
CONTEXTS_DIR: str = os.getenv("JDR_CONTEXTS_DIR", "contexts")

# Noms des fichiers de sortie
# Retranscription diarisée lisible : [HH:MM:SS] Nom : "texte"
OUTPUT_TRANSCRIPT_DIARIZED: str = "transcript_session.txt"
# Transcription brute (format interne pour chunking)
OUTPUT_TRANSCRIPT_RAW: str = "transcript_raw.txt"
# Transcription après correction LLM des noms de lore
OUTPUT_TRANSCRIPT_CORRECTED: str = "transcript_corrected.txt"
OUTPUT_GM_SHEET: str = "fiche_mj.md"
OUTPUT_PLAYER_SUMMARY: str = "resume_joueurs.md"
