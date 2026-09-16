"""
main.py — v3 : Interface CLI avec sous-commandes (subcommands).

Sous-commandes disponibles :
  build-context   Construit ou met à jour le contexte lore d'un thème
  transcribe      Lance le pipeline complet de retranscription

Exemples :
  python main.py build-context --theme "Naruto" --pdf-dir ./pdfs
  python main.py transcribe --audio session.mp3 --context "Naruto"
  python main.py transcribe --audio ./session_04/ --context "Naruto" --mapping players.json
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging (doit être configuré AVANT les imports internes)
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    """Configure le système de logging avec un format horodaté lisible."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("urllib3", "httpx", "httpcore", "pyannote", "pytorch_lightning"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Imports internes
# ---------------------------------------------------------------------------

import config
from modules.ollama_client import check_ollama_server, check_model_available
from modules.lore import build_context, load_context_for_transcription
from modules.audio import (
    transcribe_audio,
    load_player_mapping,
    format_diarized_transcript,
    prepare_audio_input,
)
from modules.chunking import build_transcript_text, chunk_transcript
from modules.summarizer import correct_all_chunks, generate_gm_sheet, generate_player_summary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Affichage console
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print(f"\n{'═' * 65}")
    print(f"  🎲  {title}")
    print(f"{'═' * 65}")


def _step(n: int, total: int, label: str) -> None:
    print(f"\n{'─' * 65}")
    print(f"  [{n}/{total}]  {label}")
    print(f"{'─' * 65}")


def _ok(msg: str) -> None:
    print(f"  ✅  {msg}")


def _warn(msg: str) -> None:
    print(f"  ⚠️   {msg}")


def _save(content: str, path: Path) -> None:
    """Sauvegarde un fichier texte UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    size_kb = path.stat().st_size / 1024
    logger.info("Fichier sauvegardé : %s (%.1f Ko)", path.name, size_kb)


# ---------------------------------------------------------------------------
# Argument parser avec sous-commandes
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retranscription-jdr",
        description=(
            "🎲  Outil de retranscription vocale pour sessions de Jeu de Rôle.\n"
            "Traitement 100%% local (WhisperX + Ollama).\n\n"
            "Sous-commandes disponibles :\n"
            "  build-context   Construit ou met à jour le contexte lore d'un thème\n"
            "  transcribe      Lance le pipeline complet de retranscription"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Variables d'environnement :\n"
            "  HF_TOKEN             Token HuggingFace (diarisation Pyannote)\n"
            "  JDR_OLLAMA_MODEL     Modèle Ollama par défaut\n"
            "  JDR_WHISPER_MODEL    Modèle Whisper par défaut\n"
            "  JDR_DEVICE           cuda / cpu\n"
        ),
    )

    # Argument global verbose
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Active les logs de débogage détaillés",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="SOUS-COMMANDE")
    subparsers.required = True

    # ------------------------------------------------------------------
    # Sous-commande : build-context
    # ------------------------------------------------------------------
    bc = subparsers.add_parser(
        "build-context",
        help="Construit ou met à jour le contexte lore depuis un dossier de PDFs.",
        description=(
            "Lit tous les PDFs d'un dossier, extrait les entités de l'univers via LLM,\n"
            "et sauvegarde (ou fusionne) un fichier JSON dans ./contexts/<theme>.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python main.py build-context --theme \"Naruto\" --pdf-dir ./pdfs\n"
            "  python main.py build-context --theme \"Fantaisie\" --pdf-dir ./lore --model llama3.1:8b\n"
        ),
    )
    bc.add_argument(
        "--theme", "-t",
        required=True,
        metavar="THEME",
        help="Nom de l'univers / thème (ex: \"Naruto\", \"Fantaisie\")",
    )
    bc.add_argument(
        "--pdf-dir", "-p",
        required=True,
        metavar="DIR",
        help="Dossier contenant les fichiers PDF de lore",
    )
    bc.add_argument(
        "--model", "-m",
        default=config.OLLAMA_MODEL,
        metavar="MODEL",
        help=f"Modèle Ollama (défaut : {config.OLLAMA_MODEL})",
    )
    bc.add_argument(
        "--contexts-dir",
        default=config.CONTEXTS_DIR,
        metavar="DIR",
        help=f"Dossier de sauvegarde des contextes (défaut : {config.CONTEXTS_DIR})",
    )

    # ------------------------------------------------------------------
    # Sous-commande : transcribe
    # ------------------------------------------------------------------
    tr = subparsers.add_parser(
        "transcribe",
        help="Lance le pipeline complet de transcription + résumé.",
        description=(
            "Transcrit un fichier audio, diarise les locuteurs, applique le contexte lore,\n"
            "génère la fiche MJ et le résumé épique pour les joueurs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  # Fichier unique\n"
            "  python main.py transcribe --audio session.mp3 --context \"Naruto\"\n"
            "  # Dossier de fichiers (multi-morceaux, tri naturel automatique)\n"
            "  python main.py transcribe --audio ./session_04/ --context \"Naruto\"\n"
            "  python main.py transcribe --audio ./session_04/ --context \"Fantaisie\" --mapping players.json\n"
            "  python main.py transcribe --audio session.mp3 --context \"Naruto\" --no-diarization\n"
            "\n"
            "Note multi-fichiers :\n"
            "  Si un dossier est fourni, tous les fichiers audio valides (.mp3, .wav, .m4a…)\n"
            "  sont triés par ordre naturel (part_1, part_2, part_10…) et concaténés\n"
            "  automatiquement via pydub avant la transcription.\n"
            "  pydub nécessite ffmpeg : winget install ffmpeg  (Windows)\n"
        ),
    )
    tr.add_argument(
        "--audio", "-a",
        required=True,
        metavar="AUDIO_OR_DIR",
        help=(
            "Fichier audio unique (mp3, wav, m4a, flac…) "
            "ou dossier contenant plusieurs fichiers audio à concaténer."
        ),
    )
    tr.add_argument(
        "--context", "-c",
        required=True,
        metavar="THEME",
        help="Thème du contexte lore à utiliser (ex: \"Naruto\")",
    )
    tr.add_argument(
        "--mapping",
        default=None,
        metavar="JSON_FILE",
        help="Fichier JSON de mapping joueurs (ex: players.json)",
    )
    tr.add_argument(
        "--model", "-m",
        default=config.OLLAMA_MODEL,
        metavar="MODEL",
        help=f"Modèle Ollama (défaut : {config.OLLAMA_MODEL})",
    )
    tr.add_argument(
        "--output-dir", "-o",
        default=config.OUTPUT_DIR,
        metavar="DIR",
        help=f"Dossier de sortie (défaut : {config.OUTPUT_DIR})",
    )
    tr.add_argument(
        "--whisper-model",
        default=config.WHISPER_MODEL,
        metavar="MODEL",
        help=f"Modèle Whisper (défaut : {config.WHISPER_MODEL})",
    )
    tr.add_argument(
        "--device",
        default=config.WHISPER_DEVICE,
        choices=["cuda", "cpu"],
        help=f"Device de calcul (défaut : {config.WHISPER_DEVICE})",
    )
    tr.add_argument(
        "--no-diarization",
        action="store_true",
        default=False,
        help="Désactive la diarisation Pyannote",
    )
    tr.add_argument(
        "--hf-token",
        default=None,
        metavar="TOKEN",
        help="Token HuggingFace (priorité sur la variable d'env HF_TOKEN)",
    )
    tr.add_argument(
        "--chunk-size",
        type=int,
        default=config.CHUNK_MAX_WORDS,
        metavar="WORDS",
        help=f"Taille max des chunks en mots (défaut : {config.CHUNK_MAX_WORDS})",
    )
    tr.add_argument(
        "--contexts-dir",
        default=config.CONTEXTS_DIR,
        metavar="DIR",
        help=f"Dossier des contextes (défaut : {config.CONTEXTS_DIR})",
    )

    return parser


# ---------------------------------------------------------------------------
# Sous-commande : build-context
# ---------------------------------------------------------------------------

TOTAL_STEPS_BUILD = 3


def run_build_context(args: argparse.Namespace) -> int:
    """
    Orchestre la construction / mise à jour du contexte lore.

    Returns:
        0 si succès, 1 si erreur critique.
    """
    start = time.time()
    theme = args.theme
    pdf_dir = Path(args.pdf_dir).resolve()
    contexts_dir = Path(args.contexts_dir).resolve()
    model = args.model

    _banner(f"BUILD-CONTEXT : thème « {theme} »")
    print(f"  Dossier PDFs  : {pdf_dir}")
    print(f"  Contextes     : {contexts_dir}")
    print(f"  Modèle LLM    : {model}")
    print(f"{'═' * 65}")

    # Étape 1 — Vérifications
    _step(1, TOTAL_STEPS_BUILD, "Vérifications préalables")
    if not pdf_dir.is_dir():
        logger.error("Dossier PDF introuvable : %s", pdf_dir)
        return 1
    try:
        check_ollama_server()
        check_model_available(model)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    _ok("Serveur Ollama actif et modèle disponible.")

    # Étape 2 — Construction / merge du contexte
    _step(2, TOTAL_STEPS_BUILD, f"Extraction et construction du contexte « {theme} »")
    try:
        context = build_context(
            theme=theme,
            pdf_dir=pdf_dir,
            contexts_dir=contexts_dir,
            model=model,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Erreur lors de la construction du contexte : %s", exc)
        return 1

    # Étape 3 — Résumé
    _step(3, TOTAL_STEPS_BUILD, "Résumé")
    from modules.lore import get_context_path
    ctx_path = get_context_path(theme, contexts_dir)
    kw_count = len(context.get("keywords", []))

    elapsed = time.time() - start
    m, s = divmod(int(elapsed), 60)

    print(f"\n{'═' * 65}")
    print(f"  🏆  CONTEXTE « {theme} » GÉNÉRÉ AVEC SUCCÈS")
    print(f"{'═' * 65}")
    print(f"  Mots-clés      : {kw_count}")
    print(f"  Personnages    : {len(context.get('characters', []))}")
    print(f"  Lieux          : {len(context.get('locations', []))}")
    print(f"  Objets         : {len(context.get('items', []))}")
    print(f"  Factions       : {len(context.get('factions', []))}")
    print(f"  Termes         : {len(context.get('terms', []))}")
    print(f"  Fichier        : {ctx_path}")
    print(f"  Durée          : {m}min {s}s")
    print(f"{'═' * 65}\n")
    return 0


# ---------------------------------------------------------------------------
# Sous-commande : transcribe
# ---------------------------------------------------------------------------

TOTAL_STEPS_TRANS = 7


def run_transcribe(args: argparse.Namespace) -> int:
    """
    Orchestre le pipeline complet de retranscription.

    Returns:
        0 si succès, 1 si erreur critique.
    """
    start = time.time()
    audio_input = Path(args.audio).resolve()
    output_dir = Path(args.output_dir).resolve()
    contexts_dir = Path(args.contexts_dir).resolve()
    theme = args.context
    model = args.model
    hf_token = args.hf_token or config.HF_TOKEN

    is_dir_input = audio_input.is_dir()
    audio_mode = "dossier multi-fichiers" if is_dir_input else "fichier unique"

    _banner(f"TRANSCRIBE : {audio_input.name} (contexte : {theme})")
    print(f"  Audio          : {audio_input}  [{audio_mode}]")
    print(f"  Contexte       : {theme}")
    print(f"  Mapping        : {args.mapping or 'Aucun'}")
    print(f"  Modèle LLM     : {model}")
    print(f"  Whisper        : {args.whisper_model} ({args.device})")
    print(f"  Diarisation    : {'NON' if args.no_diarization else 'OUI (Pyannote)'}")
    print(f"  Sortie         : {output_dir}")
    print(f"{'═' * 65}")

    # ------------------------------------------------------------------
    # Étape 1 — Vérifications préalables
    # ------------------------------------------------------------------
    _step(1, TOTAL_STEPS_TRANS, "Vérifications préalables")

    # Validation de l'entrée audio (fichier ou dossier)
    if not audio_input.exists():
        logger.error(
            "Le chemin audio n'existe pas : %s\n"
            "Fournissez un fichier audio ou un dossier contenant des fichiers audio.",
            audio_input,
        )
        return 1

    try:
        check_ollama_server()
        check_model_available(model)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    if not args.no_diarization and not hf_token:
        _warn(
            "HF_TOKEN non défini → Diarisation désactivée.\n"
            "   Fournissez-le avec : --hf-token TOKEN  ou  dans le fichier .env"
        )

    _ok("Vérifications OK.")

    # ------------------------------------------------------------------
    # Étape 2 — Chargement du contexte lore
    # ------------------------------------------------------------------
    _step(2, TOTAL_STEPS_TRANS, f"Chargement du contexte lore « {theme} »")
    try:
        lore_dict = load_context_for_transcription(theme, contexts_dir=contexts_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    _ok(f"{len(lore_dict['keywords'])} mots-clés de contexte chargés.")

    # ------------------------------------------------------------------
    # Étape 3 — Chargement du mapping joueurs (optionnel)
    # ------------------------------------------------------------------
    _step(3, TOTAL_STEPS_TRANS, "Chargement du mapping joueurs")
    player_mapping: dict | None = None
    if args.mapping:
        try:
            player_mapping = load_player_mapping(args.mapping)
            _ok(f"{len(player_mapping)} joueur(s) mappé(s).")
        except (FileNotFoundError, RuntimeError) as exc:
            logger.error("Erreur mapping : %s", exc)
            return 1
    else:
        logger.info("Aucun mapping joueurs fourni (labels Pyannote conservés).")
        _ok("Aucun mapping joueurs — labels Pyannote conservés.")

    # ------------------------------------------------------------------
    # Étape 4 — Préparation audio + Transcription
    # ------------------------------------------------------------------
    _step(4, TOTAL_STEPS_TRANS, "Préparation audio et Transcription (WhisperX + Pyannote)")
    effective_hf = None if args.no_diarization else hf_token

    try:
        with prepare_audio_input(audio_input) as (audio_path, is_merged):
            if is_merged:
                _ok(f"Fichiers audio concaténés → fichier temporaire créé.")

            segments = transcribe_audio(
                audio_path=audio_path,
                lore_dict=lore_dict,
                player_mapping=player_mapping,
                device=args.device,
                whisper_model=args.whisper_model,
                hf_token=effective_hf,
            )
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Erreur lors de la préparation ou transcription audio : %s", exc)
        return 1
    # Le fichier temporaire est supprimé ici automatiquement (fin du bloc `with`)

    if not segments:
        logger.error("Aucun segment transcrit. Vérifiez le fichier audio.")
        return 1

    _ok(f"{len(segments)} segments transcrits.")

    # ------------------------------------------------------------------
    # Étape 5 — Génération des fichiers de retranscription
    # ------------------------------------------------------------------
    _step(5, TOTAL_STEPS_TRANS, "Génération des retranscriptions")

    # 5a. Retranscription diarisée lisible (format [HH:MM:SS] Nom : "texte")
    diarized_text = format_diarized_transcript(segments)
    diarized_path = output_dir / config.OUTPUT_TRANSCRIPT_DIARIZED
    _save(diarized_text, diarized_path)
    _ok(f"Retranscription diarisée → {diarized_path.name}")

    # 5b. Transcription brute (pour le chunking LLM)
    transcript_raw = build_transcript_text(segments)
    raw_path = output_dir / config.OUTPUT_TRANSCRIPT_RAW
    _save(transcript_raw, raw_path)
    _ok(f"Transcription brute → {raw_path.name}")

    # ------------------------------------------------------------------
    # Étape 6 — Chunking et correction orthographique
    # ------------------------------------------------------------------
    _step(6, TOTAL_STEPS_TRANS, "Chunking et correction orthographique LLM")

    chunks = chunk_transcript(transcript_raw, max_words=args.chunk_size)
    logger.info("%d chunk(s) à corriger.", len(chunks))

    corrected_chunks = correct_all_chunks(chunks, lore_dict["keywords"], model=model)
    transcript_corrected = "\n\n".join(corrected_chunks)

    corrected_path = output_dir / config.OUTPUT_TRANSCRIPT_CORRECTED
    _save(transcript_corrected, corrected_path)
    _ok(f"Transcription corrigée → {corrected_path.name}")

    # ------------------------------------------------------------------
    # Étape 7 — Génération des résumés LLM
    # ------------------------------------------------------------------
    _step(7, TOTAL_STEPS_TRANS, "Génération des résumés LLM")

    gm_sheet = generate_gm_sheet(transcript_corrected, model=model)
    gm_path = output_dir / config.OUTPUT_GM_SHEET
    _save(gm_sheet, gm_path)
    _ok(f"Fiche MJ → {gm_path.name}")

    player_summary = generate_player_summary(transcript_corrected, model=model)
    summary_path = output_dir / config.OUTPUT_PLAYER_SUMMARY
    _save(player_summary, summary_path)
    _ok(f"Résumé joueurs → {summary_path.name}")

    # ------------------------------------------------------------------
    # Récapitulatif final
    # ------------------------------------------------------------------
    elapsed = time.time() - start
    m, s = divmod(int(elapsed), 60)

    output_files = [
        (config.OUTPUT_TRANSCRIPT_DIARIZED, "Retranscription diarisée"),
        (config.OUTPUT_TRANSCRIPT_RAW,       "Transcription brute"),
        (config.OUTPUT_TRANSCRIPT_CORRECTED, "Transcription corrigée"),
        (config.OUTPUT_GM_SHEET,             "Fiche MJ"),
        (config.OUTPUT_PLAYER_SUMMARY,       "Résumé joueurs"),
    ]

    print(f"\n{'═' * 65}")
    print("  🏆  PIPELINE TERMINÉ AVEC SUCCÈS")
    print(f"{'═' * 65}")
    print(f"  Durée totale   : {m}min {s}s")
    print(f"  Dossier sortie : {output_dir}")
    print()
    print("  Fichiers générés :")
    for fname, label in output_files:
        fpath = output_dir / fname
        size_kb = fpath.stat().st_size / 1024 if fpath.exists() else 0
        print(f"    📄  {label:<30} {fname:<35} ({size_kb:.1f} Ko)")
    print(f"{'═' * 65}\n")

    return 0


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    if args.command == "build-context":
        exit_code = run_build_context(args)
    elif args.command == "transcribe":
        exit_code = run_transcribe(args)
    else:
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
