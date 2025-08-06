# 🩺 Projet de Prédiction du Diabète

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)](https://www.postgresql.org/)

Une application web complète de prédiction du risque de diabète utilisant des techniques de machine learning et une interface web moderne développée avec FastAPI.

## 📑 Table des Matières

- [Aperçu du Projet](#aperçu-du-projet)
- [Architecture](#architecture)
- [Fonctionnalités](#fonctionnalités)
- [Technologies Utilisées](#technologies-utilisées)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Analyse des Données](#analyse-des-données)
- [Modèles de Machine Learning](#modèles-de-machine-learning)
- [Résultats et Performances](#résultats-et-performances)
- [Structure du Projet](#structure-du-projet)
- [API Documentation](#api-documentation)
- [Contribuer](#contribuer)
- [License](#license)

## 🎯 Aperçu du Projet

Ce projet vise à développer un système de prédiction du diabète basé sur des données cliniques. Il comprend une analyse exploratoire complète des données, l'entraînement de plusieurs modèles de machine learning, et une interface web permettant aux professionnels de santé de faire des prédictions en temps réel.

### Objectifs
- Analyser les facteurs de risque du diabète
- Développer des modèles prédictifs performants
- Créer une interface web intuitive pour les professionnels de santé
- Fournir des prédictions fiables et explicables

## 🏗️ Architecture

```
├── data/                          # Données brutes et traitées
├── notebooks/                     # Notebooks d'analyse et d'entraînement
├── models/                        # Modèles entraînés sauvegardés
├── web/                          # Application web FastAPI
│   ├── src/                      # Code source modulaire
│   │   ├── main.py              # Point d'entrée principal
│   │   ├── routes.py            # Routes de l'API
│   │   ├── utilitaires.py       # Fonctions utilitaires
│   │   └── model_pydantic.py    # Modèles de validation
│   ├── templates/               # Templates HTML
│   ├── static/                  # Fichiers statiques (CSS, JS)
│   └── .env                     # Variables d'environnement
└── requirements.txt             # Dépendances Python
```

## ✨ Fonctionnalités

### Analyse des Données
- **Exploration approfondie** : Analyse statistique descriptive
- **Visualisations interactives** : Matrices de corrélation, distributions
- **Clustering** : Segmentation des patients avec K-Means
- **Ingénierie des features** : Sélection et transformation des variables

### Modèles de Machine Learning
- **5 algorithmes testés** : Random Forest, SVM, Gradient Boosting, Decision Tree, Logistic Regression
- **Validation croisée** : Évaluation robuste des performances
- **Optimisation des hyperparamètres** : Grid Search pour chaque modèle
- **Métriques complètes** : Accuracy, Precision, Recall, F1-Score, AUC-ROC

### Interface Web
- **Authentification sécurisée** : Système de connexion avec JWT
- **Dashboard intuitif** : Statistiques et aperçu des patients
- **Gestion des patients** : CRUD complet
- **Prédictions en temps réel** : Interface de saisie et résultats instantanés
- **Design responsive** : Compatible mobile et desktop

## 🛠️ Technologies Utilisées

### Backend & ML
- **Python 3.8+** : Langage principal
- **FastAPI** : Framework web moderne et rapide
- **Scikit-Learn** : Bibliothèque de machine learning
- **Pandas & NumPy** : Manipulation et analyse des données
- **Joblib** : Sérialisation des modèles

### Base de Données
- **PostgreSQL** : Base de données relationnelle
- **psycopg2** : Connecteur Python-PostgreSQL

### Frontend & UI
- **HTML5/CSS3** : Interface utilisateur
- **Jinja2** : Moteur de templates
- **Responsive Design** : Interface adaptative

### Sécurité & Auth
- **JWT** : Authentification par tokens
- **bcrypt** : Hachage sécurisé des mots de passe
- **HTTPOnly Cookies** : Stockage sécurisé des tokens

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- PostgreSQL 13+
- Git

### 1. Cloner le Repository
```bash
git clone https://github.com/votre-username/diabete-prediction.git
cd diabete-prediction
```

### 2. Créer un Environnement Virtuel
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Installer les Dépendances
```bash
pip install -r requirements.txt
```

### 4. Configuration PostgreSQL
```sql
-- Créer une base de données
CREATE DATABASE diabete_prediction;

-- Créer un utilisateur
CREATE USER diabete_user WITH PASSWORD 'votre_mot_de_passe';
GRANT ALL PRIVILEGES ON DATABASE diabete_prediction TO diabete_user;
```

## ⚙️ Configuration

### Variables d'Environnement

Créer un fichier `.env` dans le dossier `web/` :

```env
# Base de données
DB_HOST=localhost
DB_NAME=diabete_prediction
DB_USER=diabete_user
DB_PASSWORD=votre_mot_de_passe

# Sécurité JWT
SECRET_KEY=votre_clé_secrète_très_longue_et_complexe
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_HOURS=24
```

### Génération d'une clé secrète
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 🎮 Utilisation

### 1. Entraînement des Modèles
```bash
# Exécuter les notebooks d'analyse dans l'ordre
jupyter notebook notebooks/01_exploration_analysis.ipynb
jupyter notebook notebooks/02_model_training.ipynb
```

### 2. Lancement de l'Application Web
```bash
cd web/src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Accès à l'Interface
- **URL** : http://localhost:8000
- **Créer un compte** : Inscription directe via l'interface
- **Se connecter** : Utiliser les identifiants créés

### 4. Utilisation de l'Interface

1. **Inscription/Connexion** : Créer un compte médecin
2. **Dashboard** : Vue d'ensemble des statistiques
3. **Ajouter un Patient** : Saisir les données cliniques
4. **Prédiction** : Obtenir le risque de diabète instantanément
5. **Gestion** : Consulter et gérer la liste des patients

## 📊 Analyse des Données

### Dataset
- **Source** : Pima Indians Diabetes Database
- **Échantillons** : 768 patients
- **Features** : 8 variables cliniques
- **Target** : Présence/Absence de diabète

### Variables Analysées
1. **Glucose** : Niveau de glucose plasmatique
2. **BloodPressure** : Pression artérielle diastolique
3. **SkinThickness** : Épaisseur du pli cutané
4. **Insulin** : Niveau d'insuline sérique
5. **BMI** : Indice de masse corporelle
6. **DiabetesPedigreeFunction** : Fonction héréditaire du diabète
7. **Age** : Âge du patient
8. **Pregnancies** : Nombre de grossesses

### Insights Clés
- **Corrélation forte** : Glucose et Age avec le diabète (0.544 et 0.264)
- **Facteurs de risque** : BMI, Insulin, et antécédents familiaux
- **Clustering** : Identification de 3 profils de patients distincts
- **Déséquilibre** : 65% non-diabétiques vs 35% diabétiques

## 🤖 Modèles de Machine Learning

### Algorithmes Testés

| Modèle | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| **SVM** | **98.4%** | **98.4%** | **98.4%** | **98.4%** | **99.7%** |
| Gradient Boosting | 97.7% | 97.7% | 97.7% | 97.6% | 99.2% |
| Random Forest | 96.1% | 96.3% | 96.1% | 96.0% | 98.8% |
| Decision Tree | 94.5% | 94.5% | 94.5% | 94.5% | 94.5% |
| Logistic Regression | 94.5% | 95.1% | 94.5% | 94.6% | 98.1% |

### Modèle Final Sélectionné : SVM (Support Vector Machine)

**Justification du choix :**
- ✅ **Meilleure performance globale** : 98.4% d'accuracy
- ✅ **AUC-ROC exceptionnelle** : 99.7%
- ✅ **Équilibre optimal** : Precision/Recall équilibrés
- ✅ **Robustesse** : Performance stable sur validation croisée
- ✅ **Généralisation** : Faible risque d'overfitting

### Features Utilisées dans le Modèle Final
- **Glucose** : Variable la plus prédictive
- **BMI** : Indicateur de surpoids/obésité
- **Age** : Facteur de risque croissant
- **DiabetesPedigreeFunction** : Prédisposition génétique

### Preprocessing
- **Standardisation** : StandardScaler pour normaliser les données
- **Gestion des valeurs manquantes** : Imputation par la médiane
- **Validation** : Split 80/20 train/test + validation croisée 5-fold

## 📈 Résultats et Performances

### Matrice de Confusion (SVM)
```
                Prédictions
Réalité    Non-Diabète  Diabète
Non-Diabète      91        1     (98.9% précision)
Diabète           1       35     (97.2% recall)
```

### Courbe ROC
- **AUC = 0.997** : Performance quasi-parfaite
- **Seuil optimal** : 0.5 (équilibre sensibilité/spécificité)

### Courbes d'Apprentissage
- **Convergence** : Modèle stable à partir de 300 échantillons
- **Pas d'overfitting** : Écart train-validation minimal
- **Généralisation** : Performance constante sur données inconnues

### Distribution des Probabilités
- **Séparation claire** : Faible risque vs Haut risque
- **Confiance élevée** : Prédictions majoritairement > 80% ou < 20%

## 📁 Structure du Projet

```
DIABETE_PREDICTION/
├── 📁 data/
│   ├── diabetes.csv                    # Dataset original
│   └── processed/                      # Données prétraitées
├── 📁 scripts/
│   ├── data_exploration.py       # Analyse exploratoire
│   ├── main.py   # Analyse des corrélations
│   ├── clustering_analysis.py    # Clustering K-Means
│   ├── model_training.py         # Entraînement des modèles
│   └── model_evaluation.ipynb       # Évaluation et 
comparaison
├── 📁 models/
│   └── diabetes_risk_prediction_model_v1.0.pkl  # Modèle final
├── 📁 plots/                           # Graphiques générés
├── 📁 web/
│   ├── 📁 src/
│   │   ├── main.py                     # Application FastAPI
│   │   ├── routes.py                   # Routes API
│   │   ├── utilitaires.py              # Fonctions utilitaires
│   │   └── model_pydantic.py           # Modèles de validation
│   ├── 📁 templates/                   # Templates HTML
│   │   ├── login.html
│   │   ├── register.html
│   │   ├── home.html
│   │   ├── patients.html
│   │   └── add_patient.html
│   ├── 📁 static/                      # Fichiers statiques
│   │   ├── base.css
│   │   └── login.css
│   └── .env                           # Variables d'environnement
├── requirements.txt                    # Dépendances Python
└── README.md                          # Ce fichier
```

## 📚 API Documentation

### Endpoints Principaux

#### Authentification
```http
POST /register          # Inscription d'un nouveau médecin
POST /login            # Connexion
GET  /logout           # Déconnexion
```

#### Interface Web
```http
GET  /                 # Page d'accueil (login)
GET  /home            # Dashboard principal
GET  /patients        # Liste des patients
GET  /add             # Formulaire d'ajout patient
POST /submit          # Soumission nouvelle prédiction
GET  /delete/{id}     # Suppression patient
```

### Modèles de Données

#### Patient Input
```json
{
  "name": "string",
  "age": "integer",
  "sex": "string",
  "glucose": "float",
  "bmi": "float", 
  "bloodpressure": "float",
  "pedigree": "float"
}
```

#### Prediction Response
```json
{
  "prediction": "integer (0 ou 1)",
  "probability_low_risk": "float",
  "probability_high_risk": "float",
  "risk_level": "string"
}
```

## 🔮 Améliorations Futures

### Court Terme
- [ ] **API REST** : Endpoints pour intégration externe
- [ ] **Export données** : PDF/Excel des rapports
- [ ] **Graphiques interactifs** : Visualisations patients
- [ ] **Notifications** : Alertes pour cas à risque

### Moyen Terme
- [ ] **Deep Learning** : Réseaux de neurones
- [ ] **Explainable AI** : SHAP/LIME pour interprétabilité
- [ ] **Multi-utilisateurs** : Gestion des rôles et permissions
- [ ] **Monitoring** : Tracking des performances modèle

### Long Terme
- [ ] **Données temps réel** : Intégration API médicales
- [ ] **Mobile App** : Application native
- [ ] **IA Conversationnelle** : Chatbot d'assistance
- [ ] **Prédictions multiples** : Autres pathologies

## 🤝 Contribuer

### Comment Contribuer
1. **Fork** le projet
2. **Créer** une branche feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** vos changements (`git commit -m 'Add some AmazingFeature'`)
4. **Push** vers la branche (`git push origin feature/AmazingFeature`)
5. **Ouvrir** une Pull Request

### Guidelines
- Respecter les conventions de code Python (PEP 8)
- Ajouter des tests pour nouvelles fonctionnalités
- Documenter les changements dans le README
- Utiliser des messages de commit descriptifs

### Signaler des Bugs
Utilisez les GitHub Issues avec le template :
- **Description** du problème
- **Étapes** pour reproduire
- **Comportement attendu** vs **comportement observé**
- **Environment** (OS, Python version, etc.)

## 📝 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👥 Auteurs

- **YONLI Fidèle** - Développeur Principal - [YONLI Fidèle](https://github.com/hammilidi)

## 🙏 Remerciements

- **UCI Machine Learning Repository** pour le dataset Pima Indians Diabetes
- **Scikit-Learn** pour les outils de machine learning
- **FastAPI** pour le framework web performant
- **Communauté Open Source** pour les bibliothèques utilisées

---

## 📞 Support

Pour toute question ou support :
- 📧 **Email** : yonlifidelis2@gmail.com
- 🐛 **Issues** : [GitHub Issues](https://github.com/votre-username/diabete-prediction/issues)
- 💬 **Discussions** : [GitHub Discussions](https://github.com/hammilidi/diabete-prediction/discussions)

---

**⭐ Si ce projet vous a aidé, n'oubliez pas de lui donner une étoile !**