"""
setup.py - Script d'installation automatisé multiplateforme (DevOps).

Automatise l'installation séquentielle :
  1. Création de l'environnement virtuel (.venv)
  2. Installation de PyTorch avec support CUDA
  3. Installation des dépendances (requirements.txt)
  4. Détection et test d'Ollama + pull du modèle mistral:7b-instruct
  5. Configuration et persistance du HF_TOKEN dans .env

Usage :
    python setup.py
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Couleurs ANSI pour la console
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_step(step: int, total: int, title: str) -> None:
    print(f"\n{CYAN}{'═' * 65}{RESET}")
    print(f"{BOLD}{YELLOW}  [{step}/{total}] {title}{RESET}")
    print(f"{CYAN}{'═' * 65}{RESET}")


def print_ok(msg: str) -> None:
    print(f"  {GREEN}✔ [OK]{RESET} {msg}")


def print_warn(msg: str) -> None:
    print(f"  {YELLOW}⚠ [ATTENTION]{RESET} {msg}")


def print_error(msg: str) -> None:
    print(f"  {RED}✖ [ERREUR]{RESET} {msg}")


def run_cmd(cmd_list: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd_list, check=check)


def get_venv_executables(venv_dir: Path) -> tuple[Path, Path]:
    if platform.system() == "Windows":
        py_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:
        py_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"
    return py_exe, pip_exe


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    venv_dir = root_dir / ".venv"
    total_steps = 6

    print(f"\n{BOLD}{CYAN}{'═' * 65}{RESET}")
    print(f"{BOLD}{CYAN}  🎲 RETRANSCRIPTION JDR - SCRIPT D'INSTALLATION AUTOMATISÉ 🎲{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 65}{RESET}")

    # ------------------------------------------------------------------
    # 1. Environnement virtuel
    # ------------------------------------------------------------------
    print_step(1, total_steps, "Vérification et création de l'environnement virtuel (.venv)")
    
    # Alerte sur la version Python
    v = sys.version_info
    print(f"  Python hôte détecté : {v.major}.{v.minor}.{v.micro}")
    if v.major == 3 and v.minor >= 13:
        print_warn(
            f"Python {v.major}.{v.minor} est très récent. Les bibliothèques IA (PyTorch / CTranslate2) "
            "sont optimisées pour Python 3.10 à 3.12.\n"
            "   Si une erreur de wheel survient, installez Python 3.11 (ex: winget install Python.Python.3.11)."
        )

    venv_python, venv_pip = get_venv_executables(venv_dir)
    if not venv_python.exists():
        print("  Création de l'environnement virtuel .venv...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        print_ok("Environnement .venv créé.")
    else:
        print_ok("Environnement .venv existant détecté.")

    print("  Mise à niveau des outils de packaging...")
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "--quiet"])

    # ------------------------------------------------------------------
    # 2. PyTorch avec CUDA
    # ------------------------------------------------------------------
    print_step(2, total_steps, "Installation de PyTorch avec support CUDA")
    print("  Installation de PyTorch (CUDA 11.8)...")
    res = subprocess.run([
        str(venv_pip), "install",
        "torch==2.0.1", "torchaudio==2.0.2",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ])
    if res.returncode != 0:
        print_warn("Version torch==2.0.1 indisponible pour ce binaire Python. Tentative avec le dernier build CUDA...")
        res_fallback = subprocess.run([
            str(venv_pip), "install",
            "torch", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])
        if res_fallback.returncode != 0:
            print_warn("Fallback : installation de PyTorch standard...")
            subprocess.run([str(venv_pip), "install", "torch", "torchaudio"])

    # Test CUDA
    check_code = "import torch; print(f'CUDA disponible : {torch.cuda.is_available()} | Device : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
    try:
        out = subprocess.check_output([str(venv_python), "-c", check_code], text=True).strip()
        print_ok(out)
    except Exception as e:
        print_warn(f"Vérification CUDA : {e}")

    # ------------------------------------------------------------------
    # 3. Dépendances requirements.txt
    # ------------------------------------------------------------------
    print_step(3, total_steps, "Installation du reste des dépendances (requirements.txt)")
    req_file = root_dir / "requirements.txt"
    if req_file.exists():
        subprocess.run([str(venv_pip), "install", "-r", str(req_file)])
        print_ok("Dépendances requirements.txt installées.")
    else:
        print_error("Fichier requirements.txt non trouvé.")

    # ------------------------------------------------------------------
    # 4. Vérification d'Ollama
    # ------------------------------------------------------------------
    print_step(4, total_steps, "Vérification de l'exécutable Ollama")
    ollama_path = shutil.which("ollama")
    if ollama_path:
        print_ok(f"Exécutable Ollama trouvé : {ollama_path}")
        
        # ------------------------------------------------------------------
        # 5. Téléchargement du modèle
        # ------------------------------------------------------------------
        print_step(5, total_steps, "Téléchargement du modèle LLM (mistral:7b-instruct)")
        print("  Téléchargement en cours via `ollama pull mistral:7b-instruct`...")
        pull_res = subprocess.run(["ollama", "pull", "mistral:7b-instruct"])
        if pull_res.returncode == 0:
            print_ok("Modèle mistral:7b-instruct prêt.")
        else:
            print_warn("Échec du pull Ollama. Vérifiez qu'Ollama est démarré (`ollama serve`).")
    else:
        print_warn("Ollama n'est pas encore installé ou introuvable dans le PATH.")
        print("  -> Téléchargez Ollama ici : https://ollama.com/download")
        print("  -> Lancez ensuite : ollama pull mistral:7b-instruct")
        print_step(5, total_steps, "Téléchargement du modèle (Étape ignorée car Ollama absent)")

    # ------------------------------------------------------------------
    # 6. Configuration HF_TOKEN
    # ------------------------------------------------------------------
    print_step(6, total_steps, "Configuration du Token HuggingFace (HF_TOKEN)")
    env_file = root_dir / ".env"
    existing_token = os.environ.get("HF_TOKEN", "")

    if not existing_token and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                existing_token = line.split("=", 1)[1].strip()

    if existing_token:
        print(f"  Token actuel configuré : {existing_token[:8]}...")

    print(f"\n  {CYAN}Pour la diarisation (séparation des voix), un token HuggingFace est requis.{RESET}")
    print("  Lien pour en générer un : https://huggingface.co/settings/tokens\n")
    user_token = input("  Entrez votre HF_TOKEN (Appuyez sur Entrée pour valider l'actuel ou ignorer) : ").strip()

    token_to_save = user_token if user_token else existing_token
    if token_to_save:
        os.environ["HF_TOKEN"] = token_to_save
        env_content = f"HF_TOKEN={token_to_save}\nJDR_OLLAMA_MODEL=mistral:7b-instruct\nJDR_WHISPER_MODEL=large-v2\n"
        env_file.write_text(env_content, encoding="utf-8")
        print_ok("HF_TOKEN sauvegardé dans .env et actif pour les futures exécutions.")
    else:
        print_warn("Aucun token saisi. Vous pourrez l'ajouter dans le fichier .env ultérieurement.")

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------
    print(f"\n{BOLD}{GREEN}{'═' * 65}{RESET}")
    print(f"{BOLD}{GREEN}  🎉 INSTALLATION TERMINÉE AVEC SUCCÈS ! 🎉{RESET}")
    print(f"{BOLD}{GREEN}{'═' * 65}{RESET}\n")
    print("Pour lancer une retranscription :")
    if platform.system() == "Windows":
        print(f"  {CYAN}.\\.venv\\Scripts\\python.exe main.py --audio session.mp3 --pdf lore.pdf{RESET}\n")
    else:
        print(f"  {CYAN}./.venv/bin/python main.py --audio session.mp3 --pdf lore.pdf{RESET}\n")
    print(f"Vérifiez le fichier {BOLD}ACTIONS_REQUISES.md{RESET} pour les autorisations Pyannote HuggingFace.\n")


if __name__ == "__main__":
    main()
