# 📋 Actions Manuelles Requises (Non Automatisables)

Ce document liste **strictement les actions manuelles préalables** indispensables au bon fonctionnement de l'outil et que les scripts ne peuvent pas exécuter à votre place (création de compte, acceptation de licences et installation d'Ollama).

---

## 1. 🔑 Token HuggingFace & Acceptation des Licences Pyannote

La diarisation vocale (différenciation des joueurs) utilise les modèles **Pyannote**. Ces modèles sont sous licence gated sur HuggingFace.

### Étape 1 : Créer votre compte et générer votre Token
1. Créez un compte ou connectez-vous sur [HuggingFace.co](https://huggingface.co/join).
2. Rendez-vous sur la page des tokens d'accès : **[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)**.
3. Cliquez sur **« Create new token »**, choisissez le type **« Read »** et copiez le token généré (format `hf_...`).

### Étape 2 : Accepter les conditions d'utilisation des 2 modèles
*Vous devez être connecté avec votre compte HuggingFace sur les deux pages ci-dessous et cliquer sur « Agree and access repository » :*

1. **Modèle de Diarisation Principale** :
   👉 **[https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)**

2. **Modèle de Segmentation** :
   👉 **[https://huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)**

> ℹ️ *Le script d'installation (`setup.ps1` ou `setup.py`) vous demandera simplement de coller ce token une seule fois.*

---

## 2. 🦙 Installer Ollama (LLM 100% Local)

1. Téléchargez et installez l'exécutable Windows :
   👉 **[https://ollama.com/download](https://ollama.com/download)**
2. Assurez-vous que l'application Ollama est bien lancée (icône dans la barre des tâches).

---

## 3. 🐍 Version de Python recommandée (3.10 à 3.12)

Les bibliothèques IA compilées avec CUDA (**PyTorch**, **CTranslate2 / WhisperX**) ne disposent pas encore de binaires précompilés pour les versions de développement comme Python 3.14.

Si vous avez uniquement Python 3.14 installé, installez Python 3.11 en une ligne dans votre terminal Windows :

```powershell
winget install Python.Python.3.11
```

---

## 4. 🚀 Lancement automatique

Une fois ces étapes validées, lancez simplement l'installation automatisée :

```powershell
# En PowerShell (recommandé) :
PowerShell -ExecutionPolicy Bypass -File .\setup.ps1

# Ou avec Python :
python setup.py
```
