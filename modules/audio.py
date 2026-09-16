"""
modules/audio.py — v3
Transcription audio via WhisperX avec diarisation Pyannote.

Nouveautés v3 :
  - prepare_audio_input() : accepte un fichier unique OU un dossier de fichiers audio.
    Si dossier → tri naturel + concaténation via pydub → fichier WAV temporaire.
    Le fichier temporaire est supprimé automatiquement après la transcription.
"""

import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from config import (
    WHISPER_MODEL,
    WHISPER_LANGUAGE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_BATCH_SIZE,
    HF_TOKEN,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# Segment normalisé :
# {
#   "speaker": str,    ex : "SPEAKER_00"  ou "Alice (Elfe)" après mapping
#   "start":   float,  en secondes
#   "end":     float,  en secondes
#   "text":    str,
# }
Segment = dict

# Extensions audio reconnues pour la concaténation multi-fichiers
AUDIO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus",
})


# ---------------------------------------------------------------------------
# Tri naturel (alphanumérique)
# ---------------------------------------------------------------------------

def _natural_sort_key(path: Path) -> list:
    """
    Génère une clé de tri naturel pour les noms de fichiers.
    Permet de trier 'part_2' après 'part_1' plutôt que 'part_10' avant 'part_2'.

    Exemples de tri :
        part_1.mp3, part_2.mp3, part_10.mp3  (correct)
        session_01.wav, session_02.wav        (correct)
    """
    parts = re.split(r"(\d+)", path.stem.lower())
    return [int(p) if p.isdigit() else p for p in parts]


# ---------------------------------------------------------------------------
# Concaténation multi-fichiers avec pydub
# ---------------------------------------------------------------------------

def _list_audio_files(audio_dir: Path) -> list[Path]:
    """
    Liste tous les fichiers audio valides dans un dossier, triés par ordre naturel.

    Args:
        audio_dir: Chemin du dossier audio.

    Returns:
        Liste de chemins triée naturellement.

    Raises:
        RuntimeError: Si aucun fichier audio valide n'est trouvé.
    """
    files = [
        f for f in audio_dir.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not files:
        raise RuntimeError(
            f"Aucun fichier audio ({', '.join(sorted(AUDIO_EXTENSIONS))}) "
            f"trouvé dans le dossier : {audio_dir}"
        )
    sorted_files = sorted(files, key=_natural_sort_key)
    logger.info(
        "%d fichier(s) audio détecté(s) dans %s :",
        len(sorted_files), audio_dir.name,
    )
    for i, f in enumerate(sorted_files, 1):
        logger.info("  %d. %s", i, f.name)
    return sorted_files


def _concatenate_audio_files(audio_files: list[Path], output_path: Path) -> None:
    """
    Concatène plusieurs fichiers audio en un seul fichier WAV via pydub.

    Args:
        audio_files:  Liste ordonnée de fichiers audio à fusionner.
        output_path:  Chemin du fichier WAV de sortie.

    Raises:
        RuntimeError: Si pydub ou ffmpeg n'est pas installé, ou en cas d'erreur.
    """
    try:
        from pydub import AudioSegment
        from pydub.exceptions import CouldntDecodeError
    except ImportError as exc:
        raise RuntimeError(
            "Le module 'pydub' n'est pas installé.\n"
            "Installez-le avec : pip install pydub\n"
            "⚠️  pydub nécessite également ffmpeg sur votre système :\n"
            "   Windows : winget install ffmpeg\n"
            "   Linux   : sudo apt install ffmpeg\n"
            "   macOS   : brew install ffmpeg"
        ) from exc

    logger.info("Concaténation de %d fichier(s) audio via pydub…", len(audio_files))
    combined: Optional[AudioSegment] = None

    for i, audio_file in enumerate(audio_files, 1):
        logger.info("  Chargement [%d/%d] : %s", i, len(audio_files), audio_file.name)
        try:
            segment = AudioSegment.from_file(str(audio_file))
            combined = segment if combined is None else combined + segment
        except CouldntDecodeError as exc:
            raise RuntimeError(
                f"Impossible de décoder '{audio_file.name}'.\n"
                "Vérifiez que ffmpeg est installé et que le fichier n'est pas corrompu.\n"
                f"Détail : {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Erreur lors du chargement de '{audio_file.name}' : {exc}"
            ) from exc

    if combined is None:
        raise RuntimeError("La concaténation n'a produit aucun audio.")

    duration_min = len(combined) / 1000 / 60
    logger.info(
        "Concaténation terminée : durée totale ~%.1f minutes.",
        duration_min,
    )

    logger.info("Export du fichier fusionné : %s", output_path.name)
    try:
        combined.export(str(output_path), format="wav")
    except Exception as exc:
        raise RuntimeError(f"Erreur lors de l'export WAV : {exc}") from exc

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info("Fichier WAV exporté : %s (%.1f Mo)", output_path.name, size_mb)


# ---------------------------------------------------------------------------
# Préparation de l'entrée audio (fichier unique ou dossier)
# ---------------------------------------------------------------------------

@contextmanager
def prepare_audio_input(
    audio_input: str | Path,
) -> Generator[tuple[Path, bool], None, None]:
    """
    Context manager qui prépare un fichier audio unique prêt à être traité.

    - Si `audio_input` est un **fichier** : le retourne directement, sans créer
      de fichier temporaire.
    - Si `audio_input` est un **dossier** : liste les fichiers audio valides,
      les trie naturellement, les concatène via pydub dans un fichier WAV
      temporaire, et le supprime automatiquement à la sortie du bloc `with`.

    Usage :
        with prepare_audio_input(args.audio) as (audio_path, is_merged):
            segments = transcribe_audio(audio_path, ...)

    Args:
        audio_input: Chemin vers un fichier audio ou un dossier de fichiers.

    Yields:
        Tuple (path_to_audio_file, is_merged_from_folder).
        - path_to_audio_file : le fichier à transcrire (original ou temporaire).
        - is_merged_from_folder : True si le fichier a été créé par concaténation.

    Raises:
        FileNotFoundError: Si le chemin n'existe pas.
        RuntimeError:      Si aucun audio valide, ou si la concaténation échoue.
    """
    audio_input = Path(audio_input).resolve()

    if not audio_input.exists():
        raise FileNotFoundError(
            f"Le chemin audio n'existe pas : {audio_input}\n"
            "Vérifiez que vous avez fourni un fichier ou un dossier valide."
        )

    if audio_input.is_file():
        # Cas simple : fichier unique
        if audio_input.suffix.lower() not in AUDIO_EXTENSIONS:
            logger.warning(
                "Extension '%s' inhabituelle pour un fichier audio. "
                "WhisperX tentera quand même de le lire.",
                audio_input.suffix,
            )
        logger.info("Entrée audio : fichier unique (%s)", audio_input.name)
        yield audio_input, False

    elif audio_input.is_dir():
        # Cas multi-fichiers : concaténation dans un fichier temporaire
        logger.info("Entrée audio : dossier détecté → mode multi-fichiers.")
        audio_files = _list_audio_files(audio_input)

        if len(audio_files) == 1:
            # Un seul fichier dans le dossier : inutile de concaténer
            logger.info("Un seul fichier trouvé dans le dossier, pas de concaténation nécessaire.")
            yield audio_files[0], False
            return

        # Création d'un fichier temporaire dans le dossier de sortie système
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            suffix=".wav",
            prefix="jdr_merged_session_",
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_path_str)

        try:
            _concatenate_audio_files(audio_files, tmp_path)
            yield tmp_path, True
        finally:
            # Nettoyage garanti, même en cas d'exception
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                    logger.info(
                        "Fichier temporaire supprimé : %s", tmp_path.name
                    )
                except OSError as exc:
                    logger.warning(
                        "Impossible de supprimer le fichier temporaire %s : %s",
                        tmp_path, exc,
                    )
    else:
        raise FileNotFoundError(
            f"Le chemin fourni n'est ni un fichier ni un dossier : {audio_input}"
        )


# ---------------------------------------------------------------------------
# Chargement du mapping joueurs
# ---------------------------------------------------------------------------

def load_player_mapping(mapping_path: str | Path) -> dict[str, str]:
    """
    Charge un fichier JSON de mapping joueurs.

    Format attendu :
        {
            "SPEAKER_00": "Alice (Elfe Ranger)",
            "SPEAKER_01": "Bob (Nain Guerrier)",
            "SPEAKER_02": "MJ"
        }

    Args:
        mapping_path: Chemin vers le fichier JSON.

    Returns:
        Dictionnaire {speaker_label: nom_affichable}.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        RuntimeError:      Si le JSON est invalide.
    """
    mapping_path = Path(mapping_path)
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Fichier de mapping introuvable : {mapping_path}")
    try:
        data = json.loads(mapping_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Le fichier doit contenir un objet JSON.")
        logger.info(
            "Mapping joueurs chargé : %d entrée(s) depuis %s.",
            len(data), mapping_path.name,
        )
        return {str(k): str(v) for k, v in data.items()}
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Fichier mapping invalide ({mapping_path}) : {exc}") from exc


def apply_player_mapping(
    segments: list[Segment],
    mapping: dict[str, str],
) -> list[Segment]:
    """
    Remplace les labels Pyannote (SPEAKER_XX) par les vrais noms du mapping.

    Les segments dont le speaker n'est pas dans le mapping sont conservés tels quels.

    Args:
        segments: Liste de segments diarisés.
        mapping:  Dictionnaire {speaker_label: nom_affichable}.

    Returns:
        Nouvelle liste de segments avec les noms remplacés.
    """
    if not mapping:
        return segments

    replaced = 0
    result: list[Segment] = []
    for seg in segments:
        new_seg = seg.copy()
        original = seg.get("speaker", "")
        new_seg["speaker"] = mapping.get(original, original)
        if new_seg["speaker"] != original:
            replaced += 1
        result.append(new_seg)

    logger.info(
        "Mapping appliqué : %d/%d segments mis à jour.",
        replaced, len(segments),
    )
    return result


# ---------------------------------------------------------------------------
# Formatage de la retranscription diarisée lisible
# ---------------------------------------------------------------------------

def format_diarized_transcript(segments: list[Segment]) -> str:
    """
    Formate la transcription diarisée dans un format lisible par un humain.

    Format de sortie :
        [00:12:34] Alice (Elfe Ranger) : "Nous devons partir avant l'aube."
        [00:12:52] Bob (Nain Guerrier) : "Je suis d'accord, mais on prend le chemin nord."

    Les segments consécutifs du même locuteur sont regroupés en un seul bloc.

    Args:
        segments: Liste de segments triés par heure de début.

    Returns:
        Texte formaté.
    """
    if not segments:
        return "(Aucun segment transcrit)"

    lines: list[str] = []
    current_speaker: str | None = None
    current_start: float = 0.0
    current_texts: list[str] = []

    def _flush() -> None:
        if current_speaker is not None and current_texts:
            ts = _fmt_timestamp(current_start)
            combined = " ".join(current_texts).strip()
            lines.append(f'[{ts}] {current_speaker} : "{combined}"')

    for seg in segments:
        speaker = seg.get("speaker", "INCONNU")
        text = seg.get("text", "").strip()
        start = seg.get("start", 0.0)

        if not text:
            continue

        if speaker != current_speaker:
            _flush()
            current_speaker = speaker
            current_start = start
            current_texts = [text]
        else:
            current_texts.append(text)

    _flush()
    return "\n".join(lines)


def _fmt_timestamp(seconds: float) -> str:
    """Convertit des secondes en HH:MM:SS."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Transcription + Diarisation
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio_path: str | Path,
    lore_dict: Optional[dict] = None,
    player_mapping: Optional[dict[str, str]] = None,
    device: str = WHISPER_DEVICE,
    compute_type: str = WHISPER_COMPUTE_TYPE,
    batch_size: int = WHISPER_BATCH_SIZE,
    whisper_model: str = WHISPER_MODEL,
    language: Optional[str] = WHISPER_LANGUAGE,
    hf_token: Optional[str] = HF_TOKEN,
) -> list[Segment]:
    """
    Transcrit un fichier audio et diarise les locuteurs.
    Applique optionnellement le mapping joueurs.

    Cette fonction attend un **chemin vers un fichier audio unique**.
    Pour gérer un fichier unique OU un dossier, utilisez le context manager
    `prepare_audio_input()` en amont (c'est ce que fait `run_transcribe` dans main.py).

    Args:
        audio_path:     Chemin vers le fichier audio.
        lore_dict:      Contexte lore (initial_prompt pour Whisper).
        player_mapping: Mapping {SPEAKER_XX: "Nom du joueur"} ou None.
        device:         "cuda" ou "cpu".
        compute_type:   "float16", "int8" ou "float32".
        batch_size:     Taille des batchs WhisperX.
        whisper_model:  Modèle Whisper à utiliser.
        language:       Code langue (ex: "fr") ou None pour auto-détect.
        hf_token:       Token HuggingFace pour Pyannote.

    Returns:
        Liste de segments triés, avec speaker résolu (nom du joueur si mapping fourni).

    Raises:
        FileNotFoundError: Si le fichier audio n'existe pas.
        RuntimeError:      En cas d'erreur de traitement.
    """
    try:
        import whisperx
    except ImportError as exc:
        raise RuntimeError(
            "Le module 'whisperx' n'est pas installé.\n"
            "Installez-le avec : pip install whisperx"
        ) from exc

    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")

    # Prompt initial depuis le lore
    initial_prompt: Optional[str] = None
    if lore_dict and lore_dict.get("initial_prompt"):
        initial_prompt = lore_dict["initial_prompt"]
        logger.info("Prompt Whisper initial : '%s...'", initial_prompt[:80])

    # ------------------------------------------------------------------
    # 1. Chargement de l'audio
    # ------------------------------------------------------------------
    logger.info("Chargement de l'audio : %s", audio_path.name)
    try:
        audio = whisperx.load_audio(str(audio_path))
    except Exception as exc:
        raise RuntimeError(f"Impossible de charger le fichier audio : {exc}") from exc

    # ------------------------------------------------------------------
    # 2. Transcription initiale Whisper
    # ------------------------------------------------------------------
    # Sécurité device / compute_type si CUDA n'est pas disponible
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    if device == "cuda" and not has_cuda:
        logger.warning(
            "CUDA demandé mais aucun GPU NVIDIA compatible n'est disponible. "
            "Bascule automatique sur CPU ('device=cpu', 'compute_type=int8')."
        )
        device = "cpu"
        compute_type = "int8"

    if device == "cpu" and compute_type == "float16":
        logger.info(
            "CTranslate2 ne supporte pas float16 sur CPU. Bascule automatique sur 'int8'."
        )
        compute_type = "int8"

    asr_options: dict = {}
    if initial_prompt:
        asr_options["initial_prompt"] = initial_prompt

    logger.info(
        "Chargement du modèle Whisper '%s' (%s / %s)…",
        whisper_model, device, compute_type,
    )
    try:
        model = whisperx.load_model(
            whisper_model,
            device=device,
            compute_type=compute_type,
            language=language,
            asr_options=asr_options if asr_options else None,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Impossible de charger le modèle Whisper '{whisper_model}' : {exc}"
        ) from exc

    logger.info("Transcription en cours… (peut prendre plusieurs minutes)")
    try:
        result = model.transcribe(audio, batch_size=batch_size)
    except Exception as exc:
        raise RuntimeError(f"Erreur durant la transcription Whisper : {exc}") from exc

    language_detected = result.get("language", language or "?")
    logger.info(
        "Transcription terminée. Langue : '%s'. %d segments bruts.",
        language_detected, len(result.get("segments", [])),
    )
    del model
    _free_gpu_memory()

    # ------------------------------------------------------------------
    # 3. Alignement temporel mot-à-mot
    # ------------------------------------------------------------------
    logger.info("Alignement temporel (WhisperX align)…")
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=language_detected,
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device=device,
            return_char_alignments=False,
        )
        del model_a
        _free_gpu_memory()
        logger.info("Alignement terminé.")
    except Exception as exc:
        logger.warning("Alignement échoué (%s). Transcription non-alignée utilisée.", exc)

    # ------------------------------------------------------------------
    # 4. Diarisation Pyannote
    # ------------------------------------------------------------------
    if not hf_token:
        logger.warning(
            "⚠️  Aucun HF_TOKEN fourni. Diarisation désactivée — locuteurs non identifiés."
        )
        segments = _build_segments_no_diarization(result.get("segments", []))
    else:
        logger.info("Diarisation Pyannote en cours…")
        try:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token,
                device=device,
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            segments = _build_segments(result.get("segments", []))
            logger.info("Diarisation terminée : %d segments.", len(segments))
        except Exception as exc:
            logger.warning("Diarisation échouée (%s). Locuteurs marqués 'INCONNU'.", exc)
            segments = _build_segments_no_diarization(result.get("segments", []))

    # ------------------------------------------------------------------
    # 5. Application du mapping joueurs (optionnel)
    # ------------------------------------------------------------------
    if player_mapping:
        segments = apply_player_mapping(segments, player_mapping)

    return segments


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _build_segments(raw_segments: list[dict]) -> list[Segment]:
    """Normalise les segments WhisperX avec speaker."""
    return sorted(
        [
            {
                "speaker": seg.get("speaker", "INCONNU"),
                "start": round(seg.get("start", 0.0), 2),
                "end": round(seg.get("end", 0.0), 2),
                "text": seg.get("text", "").strip(),
            }
            for seg in raw_segments
        ],
        key=lambda s: s["start"],
    )


def _build_segments_no_diarization(raw_segments: list[dict]) -> list[Segment]:
    """Construit des segments sans information de locuteur."""
    return sorted(
        [
            {
                "speaker": "LOCUTEUR",
                "start": round(seg.get("start", 0.0), 2),
                "end": round(seg.get("end", 0.0), 2),
                "text": seg.get("text", "").strip(),
            }
            for seg in raw_segments
        ],
        key=lambda s: s["start"],
    )


def _free_gpu_memory() -> None:
    """Libère la mémoire GPU si CUDA est disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
