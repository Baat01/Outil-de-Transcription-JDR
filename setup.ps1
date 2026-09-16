# ==============================================================================
# setup.ps1 - Script d'installation automatisé pour l'outil de Retranscription JDR
# ==============================================================================
# Usage :
#   PowerShell -ExecutionPolicy Bypass -File .\setup.ps1
# ==============================================================================

[CmdletBinding()]
param()

# Configuration de l'encodage de la console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Step {
    param([int]$StepNumber, [int]$TotalSteps, [string]$Title)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  [$StepNumber/$TotalSteps] $Title" -ForegroundColor Yellow
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "  [OK] $Message" -ForegroundColor Green
}

function Write-WarningMsg {
    param([string]$Message)
    Write-Host "  [ATTENTION] $Message" -ForegroundColor Yellow
}

function Write-ErrorMsg {
    param([string]$Message)
    Write-Host "  [ERREUR] $Message" -ForegroundColor Red
}

$TotalSteps = 6
$RootDir = $PSScriptRoot
if (-not $RootDir) { $RootDir = (Get-Location).Path }

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Magenta
Write-Host "   DEVOPS AUTO-INSTALLER : RETRANSCRIPTION JDR (100% LOCAL)          " -ForegroundColor Magenta
Write-Host "======================================================================" -ForegroundColor Magenta

# ------------------------------------------------------------------------------
# 1. Verification de Python et creation de l'environnement virtuel (.venv)
# ------------------------------------------------------------------------------
Write-Step 1 $TotalSteps "Verification de Python & Creation du .venv"

# Recherche d'une version compatible de Python (3.10 ou 3.11 recommandees pour PyTorch/WhisperX)
$PythonCmd = "python"
$PyLauncher = Get-Command "py" -ErrorAction SilentlyContinue

if ($PyLauncher) {
    $AvailablePy = & py --list 2>&1
    if ($AvailablePy -match "3\.11") {
        $PythonCmd = "py -3.11"
        Write-Host "  -> Python 3.11 detecte via le lanceur py (Recommande)." -ForegroundColor Cyan
    } elseif ($AvailablePy -match "3\.10") {
        $PythonCmd = "py -3.10"
        Write-Host "  -> Python 3.10 detecte via le lanceur py (Recommande)." -ForegroundColor Cyan
    } elseif ($AvailablePy -match "3\.12") {
        $PythonCmd = "py -3.12"
        Write-Host "  -> Python 3.12 detecte via le lanceur py." -ForegroundColor Cyan
    }
}

# Verification de la version actuelle
$PyVer = Invoke-Expression "$PythonCmd --version" 2>&1
Write-Host "  Version utilisee : $PyVer" -ForegroundColor White

if ($PyVer -match "3\.1[3-9]") {
    Write-WarningMsg "PyTorch et CTranslate2 (WhisperX) ont parfois des incompatibilites avec Python >= 3.13."
    Write-WarningMsg "Si l'installation echoue, installez Python 3.11 : winget install Python.Python.3.11"
}

$VenvDir = Join-Path $RootDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "  Creation de l'environnement virtuel (.venv)..." -ForegroundColor White
    Invoke-Expression "$PythonCmd -m venv `"$VenvDir`""
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPython)) {
        Write-ErrorMsg "Echec de creation du .venv. Verifiez votre installation de Python."
        exit 1
    }
    Write-Success "Environnement virtuel .venv cree avec succes."
} else {
    Write-Success "L'environnement virtuel .venv existe deja."
}

# Mise a niveau de pip
Write-Host "  Mise a jour de pip, setuptools et wheel..." -ForegroundColor White
& $VenvPython -m pip install --upgrade pip setuptools wheel --quiet

# ------------------------------------------------------------------------------
# 2. Installation prioritaire de PyTorch avec CUDA
# ------------------------------------------------------------------------------
Write-Step 2 $TotalSteps "Installation de PyTorch avec support CUDA"

Write-Host "  Installation de PyTorch (CUDA cu118)..." -ForegroundColor White
& $VenvPip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

if ($LASTEXITCODE -ne 0) {
    Write-WarningMsg "La version exacte torch==2.0.1 n'est pas disponible pour cette version de Python."
    Write-Host "  Tentative avec la version PyTorch CUDA la plus recente compatible..." -ForegroundColor Yellow
    & $VenvPip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) {
        Write-WarningMsg "Fallback : Installation de PyTorch standard..."
        & $VenvPip install torch torchaudio
    }
}

# Verification CUDA
$CudaCheck = & $VenvPython -c "import torch; print('CUDA disponible : ' + str(torch.cuda.is_available()) + ' (' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU') + ')')" 2>&1
Write-Success "$CudaCheck"

# ------------------------------------------------------------------------------
# 3. Installation du reste des dependances (requirements.txt)
# ------------------------------------------------------------------------------
Write-Step 3 $TotalSteps "Installation des dependances requirements.txt"

$ReqFile = Join-Path $RootDir "requirements.txt"
if (Test-Path $ReqFile) {
    Write-Host "  Installation de requirements.txt..." -ForegroundColor White
    & $VenvPip install -r "$ReqFile"
    if ($LASTEXITCODE -ne 0) {
        Write-WarningMsg "Certaines dependances ont signale un avertissement lors de l'installation."
    } else {
        Write-Success "Toutes les dependances sont installees."
    }
} else {
    Write-ErrorMsg "Fichier requirements.txt introuvable !"
}

# ------------------------------------------------------------------------------
# 4. Verification d'Ollama
# ------------------------------------------------------------------------------
Write-Step 4 $TotalSteps "Verification du service Ollama"

$OllamaCmd = Get-Command "ollama" -ErrorAction SilentlyContinue
if ($OllamaCmd) {
    $OllamaVersion = & ollama --version 2>&1
    Write-Success "Ollama est installe : $OllamaVersion"
    
    # --------------------------------------------------------------------------
    # 5. Telechargement du modele LLM
    # --------------------------------------------------------------------------
    Write-Step 5 $TotalSteps "Telechargement du modele LLM (mistral:7b-instruct)"
    Write-Host "  Recuperation du modele via Ollama..." -ForegroundColor White
    & ollama pull mistral:7b-instruct
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Modele mistral:7b-instruct pret a l'emploi."
    } else {
        Write-WarningMsg "Impossible de telecharger le modele. Assurez-vous qu'Ollama est lance (ollama serve)."
    }
} else {
    Write-WarningMsg "L'executable 'ollama' n'a pas ete trouve sur votre systeme."
    Write-Host "  -> Telechargez et installez Ollama depuis : https://ollama.com/download" -ForegroundColor Yellow
    Write-Host "  -> Une fois installe, lancez : ollama pull mistral:7b-instruct" -ForegroundColor Yellow
    
    Write-Step 5 $TotalSteps "Telechargement du modele LLM (Ignore - Ollama non detecte)"
}

# ------------------------------------------------------------------------------
# 6. Configuration du token HuggingFace (HF_TOKEN)
# ------------------------------------------------------------------------------
Write-Step 6 $TotalSteps "Configuration du Token HuggingFace (HF_TOKEN)"

$EnvFile = Join-Path $RootDir ".env"
$CurrentToken = $env:HF_TOKEN

if (-not $CurrentToken -and (Test-Path $EnvFile)) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match "^HF_TOKEN=(.+)$") {
            $CurrentToken = $matches[1].Trim()
        }
    }
}

if ($CurrentToken) {
    Write-Host "  Un token HuggingFace est deja configure : $($CurrentToken.Substring(0, [Math]::Min(7, $CurrentToken.Length)))..." -ForegroundColor White
}

Write-Host ""
Write-Host "  Pour la diarisation (separation des voix avec Pyannote), un token HuggingFace est requis." -ForegroundColor Cyan
Write-Host "  Lien : https://huggingface.co/settings/tokens" -ForegroundColor DarkCyan
Write-Host ""
$InputToken = Read-Host "  Entrez votre HF_TOKEN [Appuyez sur Entree pour conserver l'actuel ou ignorer]"

$TokenToSave = $CurrentToken
if ($InputToken -and $InputToken.Trim() -ne "") {
    $TokenToSave = $InputToken.Trim()
}

if ($TokenToSave) {
    $env:HF_TOKEN = $TokenToSave
    # Sauvegarde persistante dans .env
    $EnvContent = "HF_TOKEN=$TokenToSave`nJDR_OLLAMA_MODEL=mistral:7b-instruct`nJDR_WHISPER_MODEL=large-v2`n"
    Set-Content -Path $EnvFile -Value $EnvContent -Encoding UTF8
    Write-Success "HF_TOKEN configure pour la session et sauvegarde dans .env."
} else {
    Write-WarningMsg "Aucun token saisi. Vous pourrez le renseigner plus tard dans le fichier .env ou via --hf-token."
}

# ------------------------------------------------------------------------------
# Recapitulatif Final
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "   INSTALLATION TERMINEE AVEC SUCCES !                               " -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Pour lancer une retranscription :" -ForegroundColor White
Write-Host "  .\.venv\Scripts\python.exe main.py --audio votre_audio.mp3 --pdf votre_lore.pdf" -ForegroundColor Cyan
Write-Host ""
Write-Host "Consultez ACTIONS_REQUISES.md pour les validations de licence Pyannote obligatoires." -ForegroundColor Yellow
Write-Host ""
