"""
modules/chunking.py
Mise en forme de la transcription brute et découpage en chunks
pour le traitement LLM.
"""

import logging
from datetime import timedelta

from config import CHUNK_MAX_WORDS, CHUNK_OVERLAP_WORDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatage de la transcription
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    """Convertit des secondes en format [HH:MM:SS]."""
    td = timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if td.days:
        hours += td.days * 24
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def build_transcript_text(segments: list[dict]) -> str:
    """
    Formate une liste de segments en texte brut lisible avec horodatages.

    Exemple de sortie :
        [00:00:05] SPEAKER_00 : Bonjour à tous, bienvenue dans cette session.
        [00:00:12] SPEAKER_01 : Merci ! Alors, on commence par où ?

    Args:
        segments: Liste de dicts {speaker, start, end, text}.

    Returns:
        Texte formaté, prêt à être sauvegardé ou envoyé au LLM.
    """
    if not segments:
        return "(Aucun segment transcrit)"

    lines: list[str] = []
    current_speaker: str | None = None

    for seg in segments:
        speaker = seg.get("speaker", "INCONNU")
        text = seg.get("text", "").strip()
        start = seg.get("start", 0.0)

        if not text:
            continue

        # Regrouper les segments consécutifs d'un même locuteur
        if speaker != current_speaker:
            lines.append(f"\n{format_timestamp(start)} {speaker} :")
            current_speaker = speaker

        lines.append(f"  {text}")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Découpage en chunks
# ---------------------------------------------------------------------------

def chunk_transcript(
    transcript_text: str,
    max_words: int = CHUNK_MAX_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """
    Découpe un long texte de transcription en blocs de taille maximale `max_words` mots.
    Un chevauchement de `overlap_words` mots est ajouté entre deux blocs consécutifs
    pour éviter les pertes de contexte aux frontières.

    Le découpage respecte les limites de ligne (chaque segment reste intact).

    Args:
        transcript_text: La transcription brute formatée.
        max_words:       Nombre maximum de mots par chunk.
        overlap_words:   Nombre de mots de chevauchement entre chunks.

    Returns:
        Liste de chaînes de texte (chunks).
    """
    lines = transcript_text.splitlines()

    chunks: list[str] = []
    current_lines: list[str] = []
    current_word_count: int = 0
    overlap_buffer: list[str] = []

    for line in lines:
        words_in_line = len(line.split())
        current_lines.append(line)
        current_word_count += words_in_line

        if current_word_count >= max_words:
            chunk_text = "\n".join(current_lines).strip()
            if chunk_text:
                chunks.append(chunk_text)
                logger.debug("Chunk %d : %d mots.", len(chunks), current_word_count)

            # Prépare le chevauchement : on récupère les dernières lignes
            # représentant environ `overlap_words` mots
            overlap_buffer = _get_overlap_lines(current_lines, overlap_words)
            current_lines = overlap_buffer.copy()
            current_word_count = sum(len(l.split()) for l in current_lines)

    # Dernier chunk (résidu)
    if current_lines:
        remaining = "\n".join(current_lines).strip()
        if remaining:
            chunks.append(remaining)
            logger.debug("Chunk final %d : %d mots.", len(chunks), current_word_count)

    if not chunks:
        logger.warning("Aucun chunk généré. Texte vide ?")
        return [transcript_text]

    logger.info(
        "%d chunks générés (max %d mots/chunk, chevauchement %d mots).",
        len(chunks), max_words, overlap_words,
    )
    return chunks


def _get_overlap_lines(lines: list[str], overlap_words: int) -> list[str]:
    """
    Retourne les dernières lignes de `lines` dont le total de mots
    est au plus `overlap_words`.
    """
    result: list[str] = []
    count = 0
    for line in reversed(lines):
        w = len(line.split())
        if count + w > overlap_words:
            break
        result.insert(0, line)
        count += w
    return result
