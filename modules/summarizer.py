"""
modules/summarizer.py
Correction orthographique des noms de lore et génération des deux
textes de sortie : fiche MJ et résumé épique pour les joueurs.
"""

import logging
from tqdm import tqdm

from modules.ollama_client import query_llm
from config import OLLAMA_MODEL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Correction orthographique du lore dans chaque chunk
# ---------------------------------------------------------------------------

_SYSTEM_CORRECTOR = (
    "Tu es un correcteur expert en jeux de rôle. "
    "Tu corriges UNIQUEMENT les fautes d'orthographe concernant les noms propres et termes spécifiques "
    "listés dans le dictionnaire de lore fourni. "
    "Tu ne modifies PAS la ponctuation, les horodatages, les noms de locuteurs ni le style narratif. "
    "Tu restitues le texte corrigé dans son intégralité, sans commentaire."
)

_PROMPT_CORRECTOR_TEMPLATE = """\
Dictionnaire de lore (noms propres corrects) :
{keywords_list}

Transcription à corriger :
---
{chunk}
---

Retranscription corrigée (sans commentaire, texte intégral) :
"""


def correct_lore_spelling(
    chunk: str,
    lore_keywords: list[str],
    model: str = OLLAMA_MODEL,
) -> str:
    """
    Corrige les noms propres d'un chunk de transcription selon le dictionnaire de lore.

    Args:
        chunk:          Bloc de texte brut à corriger.
        lore_keywords:  Liste des mots-clés corrects extraits du PDF.
        model:          Modèle Ollama à utiliser.

    Returns:
        Le chunk avec les noms corrigés.
    """
    if not lore_keywords:
        logger.debug("Pas de mots-clés de lore — chunk retourné sans correction.")
        return chunk

    keywords_list = "\n".join(f"- {kw}" for kw in lore_keywords[:150])
    prompt = _PROMPT_CORRECTOR_TEMPLATE.format(
        keywords_list=keywords_list,
        chunk=chunk,
    )

    try:
        corrected = query_llm(prompt, system=_SYSTEM_CORRECTOR, model=model, temperature=0.1)
        return corrected if corrected.strip() else chunk
    except RuntimeError as exc:
        logger.warning("Correction LLM échouée pour ce chunk (%s). Original conservé.", exc)
        return chunk


def correct_all_chunks(
    chunks: list[str],
    lore_keywords: list[str],
    model: str = OLLAMA_MODEL,
) -> list[str]:
    """
    Corrige tous les chunks de transcription avec une barre de progression.

    Args:
        chunks:        Liste des chunks bruts.
        lore_keywords: Mots-clés de lore.
        model:         Modèle Ollama.

    Returns:
        Liste des chunks corrigés dans le même ordre.
    """
    logger.info("Correction orthographique de %d chunk(s) via LLM…", len(chunks))
    corrected_chunks: list[str] = []

    for i, chunk in enumerate(
        tqdm(chunks, desc="  Correction chunks", unit="chunk", ncols=80), start=1
    ):
        logger.debug("Correction chunk %d/%d…", i, len(chunks))
        corrected = correct_lore_spelling(chunk, lore_keywords, model=model)
        corrected_chunks.append(corrected)

    logger.info("Correction terminée.")
    return corrected_chunks


# ---------------------------------------------------------------------------
# Fiche Maître du Jeu
# ---------------------------------------------------------------------------

_SYSTEM_GM = (
    "Tu es un assistant pour Maître du Jeu de jeu de rôle sur table. "
    "Tu analyses des transcriptions de sessions et tu produis des fiches structurées, précises et exploitables. "
    "Réponds en Markdown."
)

_PROMPT_GM_TEMPLATE = """\
Voici la transcription complète (ou partielle) d'une session de jeu de rôle :

---
{transcript}
---

Génère une **fiche récapitulative pour le Maître du Jeu** au format Markdown avec les sections suivantes :

## 🧑‍🤝‍🧑 Personnages Non-Joueurs (PNJ)
Liste tous les PNJ mentionnés, avec une brève description si possible.

## 📦 Objets et Lieux Importants
Objets magiques, artefacts, lieux clés évoqués durant la session.

## 🕵️ Fils Narratifs et Intrigues
Points importants de l'intrigue : décisions des joueurs, rebondissements, secrets révélés.

## ✅ Résumé des Actions des Joueurs
Ce que les joueurs ont accompli concrètement durant la session.

## 📋 Notes pour la prochaine session
Éléments en suspens, hooks à relancer, questions laissées ouvertes.
"""


def generate_gm_sheet(
    full_transcript: str,
    model: str = OLLAMA_MODEL,
    max_chars: int = 14_000,
) -> str:
    """
    Génère une fiche récapitulative Markdown pour le Maître du Jeu.

    Args:
        full_transcript: Transcription corrigée complète.
        model:           Modèle Ollama.
        max_chars:       Troncature de sécurité si la transcription est très longue.

    Returns:
        Texte Markdown de la fiche MJ.
    """
    logger.info("Génération de la fiche MJ…")

    transcript_excerpt = _truncate(full_transcript, max_chars)
    prompt = _PROMPT_GM_TEMPLATE.format(transcript=transcript_excerpt)

    try:
        gm_sheet = query_llm(prompt, system=_SYSTEM_GM, model=model, temperature=0.3)
        logger.info("Fiche MJ générée (%d caractères).", len(gm_sheet))
        return gm_sheet
    except RuntimeError as exc:
        logger.error("Impossible de générer la fiche MJ : %s", exc)
        return f"# Erreur de génération\n\n{exc}"


# ---------------------------------------------------------------------------
# Résumé épique pour les joueurs (style Dragon Ball Z)
# ---------------------------------------------------------------------------

_SYSTEM_PLAYER = (
    "Tu es un narrateur épique qui résume des sessions de jeu de rôle avec la grandiloquence "
    "et le souffle dramatique des introductions de Dragon Ball Z. "
    "Ton texte est court, percutant, vibrant d'énergie. "
    "Tu parles au présent, à la deuxième personne du pluriel (« vous »), "
    "comme si tu annonçais un épisode légendaire. "
    "Réponds en Français."
)

_PROMPT_PLAYER_TEMPLATE = """\
Voici la transcription d'une session de jeu de rôle :

---
{transcript}
---

Écris un **résumé épique de 150 mots MAXIMUM** pour les joueurs, dans le style des introductions de Dragon Ball Z :
- Commence par une phrase d'accroche dramatique.
- Résume les événements clés de manière héroïque et exaltante.
- Mentionne les noms des personnages joueurs si identifiables.
- Termine par une phrase de cliffhanger ou de promesse pour la prochaine session.
- Style : présent, deuxième personne du pluriel (« vous »), phrases courtes et puissantes.

Résumé (150 mots max) :
"""


def generate_player_summary(
    full_transcript: str,
    model: str = OLLAMA_MODEL,
    max_chars: int = 10_000,
) -> str:
    """
    Génère un court résumé épique style Dragon Ball Z pour les joueurs.

    Args:
        full_transcript: Transcription corrigée complète.
        model:           Modèle Ollama.
        max_chars:       Troncature de sécurité.

    Returns:
        Texte du résumé épique (≤ 150 mots).
    """
    logger.info("Génération du résumé épique pour les joueurs…")

    transcript_excerpt = _truncate(full_transcript, max_chars)
    prompt = _PROMPT_PLAYER_TEMPLATE.format(transcript=transcript_excerpt)

    try:
        summary = query_llm(prompt, system=_SYSTEM_PLAYER, model=model, temperature=0.7)
        word_count = len(summary.split())
        logger.info("Résumé joueurs généré : %d mots.", word_count)
        return summary
    except RuntimeError as exc:
        logger.error("Impossible de générer le résumé joueurs : %s", exc)
        return f"Erreur de génération : {exc}"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int) -> str:
    """Tronque intelligemment un texte trop long pour le LLM."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    truncated = text[:half] + "\n\n[... TRANSCRIPTION TRONQUÉE ...]\n\n" + text[-half:]
    logger.debug("Transcription tronquée : %d → ~%d caractères.", len(text), max_chars)
    return truncated
