"""
modules/lore.py — v2
Gestion du contexte (Lore) multi-PDF et thématique.

Nouvelles fonctionnalités :
  - Lecture de tous les PDF d'un dossier (--pdf-dir)
  - Sauvegarde du contexte dans ./contexts/<theme>.json
  - Merge intelligent avec le contexte existant via LLM
  - Construction du prompt initial WhisperX à partir du contexte JSON
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from modules.ollama_client import query_llm
from config import OLLAMA_MODEL, CONTEXTS_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schéma du fichier de contexte JSON
# ---------------------------------------------------------------------------
# {
#   "theme": "Naruto",
#   "updated_at": "2026-09-16T10:00:00",
#   "characters": ["Naruto", "Sasuke", ...],
#   "locations": ["Konoha", "Village de la Feuille", ...],
#   "items": ["Kunai", "Scroll des Techniques Interdites", ...],
#   "factions": ["L'Akatsuki", "Les Sannin", ...],
#   "terms": ["Chakra", "Sharingan", "Jutsu", ...],
#   "keywords": [...]   <- union plate de tout ce qui précède, pour WhisperX
# }

CONTEXT_SCHEMA_KEYS = ["characters", "locations", "items", "factions", "terms"]


# ---------------------------------------------------------------------------
# Chemins des contextes
# ---------------------------------------------------------------------------

def get_context_path(theme: str, contexts_dir: str | Path = CONTEXTS_DIR) -> Path:
    """Retourne le chemin du fichier JSON pour un thème donné."""
    sanitized = theme.strip().lower().replace(" ", "_")
    return Path(contexts_dir) / f"{sanitized}.json"


def load_context(theme: str, contexts_dir: str | Path = CONTEXTS_DIR) -> dict:
    """
    Charge le fichier de contexte JSON existant pour un thème.
    Retourne un contexte vide si le fichier n'existe pas.
    """
    path = get_context_path(theme, contexts_dir)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            logger.info("Contexte existant chargé : %s (%d mots-clés)", path.name, len(data.get("keywords", [])))
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Impossible de lire le contexte existant (%s). Nouveau contexte créé.", exc)
    return _empty_context(theme)


def save_context(context: dict, theme: str, contexts_dir: str | Path = CONTEXTS_DIR) -> Path:
    """
    Sauvegarde le contexte dans ./contexts/<theme>.json.
    Crée le dossier si nécessaire.
    """
    path = get_context_path(theme, contexts_dir)
    Path(contexts_dir).mkdir(parents=True, exist_ok=True)
    context["updated_at"] = datetime.now().isoformat(timespec="seconds")
    context["theme"] = theme
    # Reconstruire la liste plate keywords pour WhisperX
    context["keywords"] = _build_flat_keywords(context)
    path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Contexte sauvegardé : %s", path)
    return path


def _empty_context(theme: str) -> dict:
    return {
        "theme": theme,
        "updated_at": "",
        "characters": [],
        "locations": [],
        "items": [],
        "factions": [],
        "terms": [],
        "keywords": [],
    }


def _build_flat_keywords(context: dict) -> list[str]:
    """Construit la liste plate dédupliquée de tous les mots-clés du contexte."""
    seen: set[str] = set()
    result: list[str] = []
    for key in CONTEXT_SCHEMA_KEYS:
        for kw in context.get(key, []):
            if kw.lower() not in seen:
                seen.add(kw.lower())
                result.append(kw)
    return result


# ---------------------------------------------------------------------------
# Extraction PDF
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extrait tout le texte d'un fichier PDF via PyMuPDF.

    Raises:
        FileNotFoundError: Si le fichier PDF n'existe pas.
        RuntimeError:      En cas d'erreur de lecture.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Fichier PDF introuvable : {pdf_path}")

    logger.info("Lecture du PDF : %s", pdf_path.name)
    try:
        doc = fitz.open(str(pdf_path))
        pages_text = [doc.load_page(i).get_text("text") for i in range(len(doc))]
        doc.close()
        full_text = "\n".join(pages_text)
        logger.info("PDF lu : %d pages, %d caractères.", len(pages_text), len(full_text))
        return full_text
    except fitz.FileDataError as exc:
        raise RuntimeError(f"PDF corrompu ou invalide : {pdf_path}\nDétail : {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Erreur lecture PDF : {exc}") from exc


def extract_texts_from_pdf_dir(pdf_dir: str | Path) -> list[tuple[str, str]]:
    """
    Lit tous les PDF d'un dossier.

    Returns:
        Liste de tuples (nom_du_fichier, texte_extrait).

    Raises:
        FileNotFoundError: Si le dossier n'existe pas.
        RuntimeError:      Si aucun PDF n'est trouvé.
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"Dossier PDF introuvable : {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"Aucun fichier PDF trouvé dans : {pdf_dir}")

    logger.info("%d PDF trouvé(s) dans %s.", len(pdf_files), pdf_dir)
    results: list[tuple[str, str]] = []
    for pdf_path in pdf_files:
        try:
            text = extract_text_from_pdf(pdf_path)
            results.append((pdf_path.name, text))
        except RuntimeError as exc:
            logger.warning("PDF ignoré (%s) : %s", pdf_path.name, exc)

    if not results:
        raise RuntimeError("Aucun PDF lisible dans le dossier.")
    return results


# ---------------------------------------------------------------------------
# Prompts LLM pour l'extraction structurée
# ---------------------------------------------------------------------------

_SYSTEM_EXTRACTOR = (
    "Tu es un assistant expert en analyse de documents de jeu de rôle et de fiction. "
    "Tu extrais et catégorises précisément les éléments de l'univers fictif. "
    "Réponds UNIQUEMENT en JSON valide, sans markdown ni commentaire."
)

_PROMPT_EXTRACT_TEMPLATE = """\
Univers / Thème : "{theme}"

Voici un extrait de document de lore :
---
{text_excerpt}
---

Extrais tous les éléments pertinents et retourne un objet JSON avec exactement ces clés :
{{
  "characters": ["liste des noms de personnages, PNJ, divinités"],
  "locations": ["villes, régions, donjons, plans, lieux importants"],
  "items": ["objets magiques, artefacts, équipements notables"],
  "factions": ["factions, guildes, organisations, clans"],
  "terms": ["termes techniques, sorts, races, classes, titres, concepts spécifiques à cet univers"]
}}

Règles :
- N'invente rien, extrais uniquement ce qui est dans le texte.
- Respecte l'orthographe exacte des noms.
- Si une catégorie est vide, mets un tableau vide [].
- Réponds avec le JSON uniquement, sans balise markdown.
"""

_SYSTEM_MERGER = (
    "Tu es un assistant expert en gestion de bases de données de lore pour jeux de rôle. "
    "Tu fusionnes intelligemment deux ensembles de données JSON en évitant les doublons. "
    "Réponds UNIQUEMENT en JSON valide, sans markdown ni commentaire."
)

_PROMPT_MERGE_TEMPLATE = """\
Univers / Thème : "{theme}"

Contexte existant (JSON) :
{existing_json}

Nouvelles données extraites (JSON) :
{new_json}

Fusionne ces deux objets JSON en un seul, en respectant ces règles :
1. Conserve TOUS les éléments existants (ne supprime rien).
2. Ajoute les nouveaux éléments manquants dans chaque catégorie.
3. Déduplique en ignorant la casse (garde la version avec la meilleure orthographe).
4. Trie chaque liste par ordre alphabétique.
5. Conserve exactement les mêmes clés : characters, locations, items, factions, terms.
6. Réponds avec le JSON uniquement, sans balise markdown.
"""


# ---------------------------------------------------------------------------
# Extraction depuis texte brut
# ---------------------------------------------------------------------------

def _extract_structured_from_text(
    text: str,
    theme: str,
    model: str = OLLAMA_MODEL,
    max_chars: int = 12_000,
) -> dict:
    """Demande au LLM d'extraire un objet JSON structuré depuis un texte."""
    if len(text) > max_chars:
        half = max_chars // 2
        excerpt = text[:half] + "\n[...]\n" + text[-half:]
    else:
        excerpt = text

    prompt = _PROMPT_EXTRACT_TEMPLATE.format(theme=theme, text_excerpt=excerpt)
    try:
        raw = query_llm(prompt, system=_SYSTEM_EXTRACTOR, model=model, temperature=0.1)
        # Nettoyage des balises markdown éventuelles
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        # Normalisation des clés
        return {k: data.get(k, []) for k in CONTEXT_SCHEMA_KEYS}
    except (json.JSONDecodeError, RuntimeError) as exc:
        logger.warning("Extraction LLM échouée pour ce PDF (%s). Fragment ignoré.", exc)
        return {k: [] for k in CONTEXT_SCHEMA_KEYS}


# ---------------------------------------------------------------------------
# Merge via LLM
# ---------------------------------------------------------------------------

def _merge_contexts_with_llm(
    existing: dict,
    new_data: dict,
    theme: str,
    model: str = OLLAMA_MODEL,
) -> dict:
    """
    Utilise le LLM pour fusionner l'ancien contexte avec les nouvelles données.
    Fallback Python si le LLM échoue.
    """
    existing_sub = {k: existing.get(k, []) for k in CONTEXT_SCHEMA_KEYS}
    new_sub = {k: new_data.get(k, []) for k in CONTEXT_SCHEMA_KEYS}

    prompt = _PROMPT_MERGE_TEMPLATE.format(
        theme=theme,
        existing_json=json.dumps(existing_sub, ensure_ascii=False, indent=2),
        new_json=json.dumps(new_sub, ensure_ascii=False, indent=2),
    )
    try:
        raw = query_llm(prompt, system=_SYSTEM_MERGER, model=model, temperature=0.1)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        merged = json.loads(raw)
        return {k: merged.get(k, []) for k in CONTEXT_SCHEMA_KEYS}
    except (json.JSONDecodeError, RuntimeError) as exc:
        logger.warning("Merge LLM échoué (%s). Fallback merge Python.", exc)
        return _merge_contexts_python(existing_sub, new_sub)


def _merge_contexts_python(existing: dict, new_data: dict) -> dict:
    """Fallback : merge simple par union dédupliquée, sans LLM."""
    merged = {}
    for key in CONTEXT_SCHEMA_KEYS:
        seen: set[str] = set()
        combined: list[str] = []
        for item in existing.get(key, []) + new_data.get(key, []):
            if item.lower() not in seen:
                seen.add(item.lower())
                combined.append(item)
        merged[key] = sorted(combined)
    return merged


# ---------------------------------------------------------------------------
# Fonction principale : build_context
# ---------------------------------------------------------------------------

def build_context(
    theme: str,
    pdf_dir: str | Path,
    contexts_dir: str | Path = CONTEXTS_DIR,
    model: str = OLLAMA_MODEL,
) -> dict:
    """
    Construit ou met à jour le contexte JSON pour un thème donné.

    1. Charge le contexte existant (s'il existe).
    2. Lit tous les PDF du dossier.
    3. Extrait les entités de chaque PDF via LLM.
    4. Fusionne avec le contexte existant via LLM (merge intelligent).
    5. Sauvegarde le résultat dans ./contexts/<theme>.json.

    Args:
        theme:        Nom de l'univers / thème (ex: "Naruto", "Fantaisie").
        pdf_dir:      Dossier contenant les PDFs à analyser.
        contexts_dir: Dossier de sauvegarde des contextes.
        model:        Modèle Ollama.

    Returns:
        Le contexte dict mis à jour avec "keywords" pour WhisperX.
    """
    logger.info("Construction du contexte pour le thème : '%s'", theme)

    # Chargement du contexte existant
    existing_context = load_context(theme, contexts_dir)
    had_existing = bool(existing_context.get("keywords"))

    # Lecture de tous les PDFs
    pdf_texts = extract_texts_from_pdf_dir(pdf_dir)
    logger.info("Traitement de %d PDF(s) pour le thème '%s'...", len(pdf_texts), theme)

    # Extraction structurée de chaque PDF
    aggregated: dict = {k: [] for k in CONTEXT_SCHEMA_KEYS}
    for pdf_name, text in pdf_texts:
        logger.info("  -> Extraction depuis : %s", pdf_name)
        extracted = _extract_structured_from_text(text, theme=theme, model=model)
        # Agrégation locale (simple merge Python)
        for key in CONTEXT_SCHEMA_KEYS:
            aggregated[key].extend(extracted.get(key, []))

    # Merge avec le contexte existant
    if had_existing:
        logger.info("Fusion avec le contexte existant via LLM...")
        merged_data = _merge_contexts_with_llm(existing_context, aggregated, theme=theme, model=model)
    else:
        logger.info("Aucun contexte existant. Déduplication simple...")
        merged_data = _merge_contexts_python({}, aggregated)

    # Construction de l'objet contexte final
    final_context = existing_context.copy()
    final_context.update(merged_data)

    # Sauvegarde
    context_path = save_context(final_context, theme, contexts_dir)
    total_kw = len(final_context.get("keywords", []))
    logger.info(
        "Contexte '%s' finalisé : %d mots-clés au total → %s",
        theme, total_kw, context_path.name,
    )
    return final_context


# ---------------------------------------------------------------------------
# Chargement du contexte pour la transcription
# ---------------------------------------------------------------------------

def load_context_for_transcription(
    theme: str,
    contexts_dir: str | Path = CONTEXTS_DIR,
) -> dict:
    """
    Charge le contexte JSON d'un thème existant et construit le prompt WhisperX.

    Returns:
        dict avec "keywords" (list) et "initial_prompt" (str).

    Raises:
        FileNotFoundError: Si le contexte n'existe pas pour ce thème.
    """
    path = get_context_path(theme, contexts_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Aucun contexte trouvé pour le thème '{theme}'.\n"
            f"Fichier attendu : {path}\n"
            f"Créez-le d'abord avec : python main.py build-context --theme \"{theme}\" --pdf-dir ./pdfs"
        )

    context = load_context(theme, contexts_dir)
    keywords = context.get("keywords", [])
    initial_prompt = ", ".join(keywords[:200])  # Whisper accepte ~224 tokens
    return {
        "keywords": keywords,
        "initial_prompt": initial_prompt,
        "context": context,
    }
