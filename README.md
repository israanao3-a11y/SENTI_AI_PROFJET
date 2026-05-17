# 👁️ Real-Time Gaze Tracking & Emotion Recognition pipeline

Un pipeline unifié et optimisé en temps réel combinant l'Apprentissage Profond (CNN) pour la reconnaissance des expressions faciales et l'analyse géométrique topologique pour le suivi du regard (*Gaze Tracking*).

---

## 📌 Présentation du Projet
Ce projet implémente une solution logicielle innovante d'analyse comportementale et d'interaction homme-machine (IHM). À partir du flux d'une simple caméra, l'application capture en direct les expressions faciales afin de prédire l'état émotionnel, tout en suivant précisément l'orientation tridimensionnelle du regard.

## 🚀 Fonctionnalités Principales
* **Classification Affective (CNN) :** Modèle basé sur l'architecture de réseaux de neurones convolutifs, entraîné sur la base standardisée **FER2013** (7 émotions détectées).
* **Suivi du Regard (Gaze Tracking) :** Extraction des repères oculaires et de l'iris en temps réel via la solution de pointe **MediaPipe Face Mesh**.
* **Filtre de Stabilisation Algorithmique :** Intégration d'une moyenne glissante (*Moving Average*) sur l'historique des trames pour éliminer les sauts de pourcentages et fluidifier l'interface graphique.
* **Extraction et Analyse de Données (Logging) :** Journalisation automatisée et transparente des métriques attentionnelles dans un fichier structuré `gaze_data.csv`.

---

## 📂 Structure du Répertoire

| Fichier / Dossier | Description |
| :--- | :--- |
| `main_tracker.py` | Script principal orchestrant le traitement vidéo et l'interface graphique utilisateur. |
| `gaze_logger.py` | Module utilitaire dédié à l'écriture et à la sauvegarde des données comportementales. |
| `gaze_tracker_utils.py` | Fonctions mathématiques et géométriques calculant le vecteur du regard. |
| `gaze_visualizer.py` | Gestionnaire des superpositions graphiques et de l'affichage des textes à l'écran. |
| `fer2013_emotion_model.h5` | Fichier de poids compressé du modèle d'apprentissage profond. |
| `gaze_data.csv` | Fichier de logs généré automatiquement (comptage des fixations et des switchs). |

---

## 🛠️ Installation et QuickStart

### 1. Cloner le projet & Activer l'environnement
```bash
# Activation de la structure virtuelle locale (Windows)
.\env311\Scripts\activate
