# Instructions pour publier sur GitHub

La GitHub CLI (`gh`) n'étant pas installée ou configurée sur votre machine, voici la procédure rapide pour publier votre dépôt sur GitHub :

---

### Étape 1 : Créer le dépôt sur GitHub

1. Rendez-vous sur [https://github.com/new](https://github.com/new).
2. Renseignez les paramètres suivants :
   - **Repository name** : `Outil-de-Transcription-JDR`
   - **Description** (facultatif) : `Outil local de retranscription, diarisation et résumé pour parties de Jeu de Rôle.`
   - **Visibility** : `Public` (ou `Private` selon votre préférence).
   - ⚠️ **Important** : Ne cochez **PAS** les cases *"Add a README file"*, *"Add .gitignore"* ou *"Choose a license"* (votre projet contient déjà ces éléments).
3. Cliquez sur **Create repository**.

---

### Étape 2 : Lier le dépôt distant et pousser le code

Ouvrez votre terminal dans le dossier du projet (`c:\Users\bruellan\Documents\Project\Training\Retranscription JDR`) et exécutez les commandes suivantes :

```powershell
# 1. Associer l'URL distante (remplacez VOTRE_NOM_UTILISATEUR par votre pseudo GitHub)
git remote add origin https://github.com/VOTRE_NOM_UTILISATEUR/Outil-de-Transcription-JDR.git

# 2. S'assurer d'être sur la branche principale
git branch -M main

# 3. Pousser le code et configurer le suivi amont
git push -u origin main
```

*(Si vous utilisez SSH plutôt que HTTPS, utilisez l'URL `git@github.com:VOTRE_NOM_UTILISATEUR/Outil-de-Transcription-JDR.git`)*.
