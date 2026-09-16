"""
modules/ollama_client.py
Wrapper autour du client Python Ollama.
Fournit un health-check et une fonction de requête générique.
"""

import logging
from typing import Optional

import ollama

from config import OLLAMA_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_ollama_server() -> None:
    """
    Vérifie que le serveur Ollama est actif et joignable.
    Lève une RuntimeError si ce n'est pas le cas.
    """
    try:
        models = ollama.list()
        model_names = [m.model for m in models.models]
        logger.info("Serveur Ollama actif. Modèles disponibles : %s", ", ".join(model_names) or "(aucun)")
    except Exception as exc:
        raise RuntimeError(
            "Impossible de joindre le serveur Ollama. "
            "Assurez-vous qu'il est lancé avec la commande : `ollama serve`\n"
            f"Détail : {exc}"
        ) from exc


def check_model_available(model: str = OLLAMA_MODEL) -> None:
    """
    Vérifie que le modèle demandé est présent localement.
    Lève une RuntimeError avec les instructions de téléchargement si absent.
    """
    try:
        models = ollama.list()
        model_names = [m.model for m in models.models]
        # Normalisation : ollama peut stocker "mistral:7b-instruct" ou "mistral:7b-instruct-v0.1"
        base_names = [n.split(":")[0] for n in model_names]
        requested_base = model.split(":")[0]

        if model not in model_names and requested_base not in base_names:
            raise RuntimeError(
                f"Le modèle '{model}' n'est pas installé localement.\n"
                f"Installez-le avec : `ollama pull {model}`\n"
                f"Modèles disponibles : {', '.join(model_names) or '(aucun)'}"
            )
        logger.info("Modèle Ollama '%s' disponible.", model)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Erreur lors de la vérification du modèle : {exc}") from exc


# ---------------------------------------------------------------------------
# Requête LLM
# ---------------------------------------------------------------------------

def query_llm(
    prompt: str,
    system: Optional[str] = None,
    model: str = OLLAMA_MODEL,
    temperature: float = 0.2,
) -> str:
    """
    Envoie un prompt au modèle Ollama et retourne la réponse sous forme de chaîne.

    Args:
        prompt:      Le message utilisateur.
        system:      Optionnel – instruction système pour définir le rôle du LLM.
        model:       Nom du modèle Ollama à utiliser.
        temperature: Créativité du modèle (faible = plus déterministe).

    Returns:
        Le contenu texte de la réponse du LLM.

    Raises:
        RuntimeError: En cas d'échec de la requête.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        logger.debug("Requête LLM (%s) — %d caractères dans le prompt.", model, len(prompt))
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
        )
        content: str = response.message.content
        logger.debug("Réponse LLM reçue — %d caractères.", len(content))
        return content.strip()
    except Exception as exc:
        raise RuntimeError(f"Erreur lors de la requête Ollama : {exc}") from exc
