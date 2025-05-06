# 🧠 Projet de Scoring Crédit - "Prêt à dépenser"

Ce projet a pour objectif de prédire si un client est un bon ou un mauvais payeur à partir de ses données personnelles, professionnelles et de crédit, en s'appuyant sur les données publiques de Home Credit.  
Il comprend une **API FastAPI** et un **dashboard Streamlit** permettant de charger les fichiers, faire une prédiction, visualiser des explications SHAP et explorer des indicateurs clés.

---

## 🚀 Fonctionnalités principales

- 🔍 Traitement des données (`preprocessing.py`, `feature_engineering.py`)
- 📊 Entraînement d’un modèle LightGBM optimisé avec SMOTE
- 🧪 Calcul d’un **score métier** prenant en compte les faux positifs et faux négatifs
- 🧠 Explicabilité via SHAP (globale et locale)
- 🌐 **API FastAPI** pour exposer le modèle
- 🖥️ **Dashboard Streamlit** pour permettre aux chargés de clientèle de :
  - Uploader les fichiers
  - Sélectionner un client
  - Obtenir une prédiction ("accorder" ou "refuser")
  - Visualiser les explications SHAP
  - Consulter des indicateurs (âge, type de travail...)

---

## 📁 Structure du projet

credit_score_projet/
│
├── api/ # Application FastAPI (main.py)
│
├── dashboard/ # Application Streamlit (app.py)
│
├── data/
│ ├── original/ # Jeux de données initiaux (non suivis par Git)
│ └── modified/ # Fichiers transformés (ex: df_final.csv)
│
├── models/ # Modèle entraîné et fichiers auxiliaires
│ ├── best_model_lightgbm.pkl
│ ├── columns_used.pkl
│ └── columns_dtypes.pkl
│
├── notebook/ # Notebook d'entraînement et d'analyse
│ └── notebook_credit_score.ipynb
│
├── src/ # Fonctions de preprocessing et feature engineering
│ ├── preprocessing.py
│ └── feature_engineering.py
│
├── tests/ # Tests unitaires (à compléter)
│
├── .gitignore # Fichiers et dossiers exclus de Git
├── requirements.txt # Dépendances du projet
└── README.md # Ce fichier


---

## ⚙️ Lancer le projet localement

### 1. Créer un environnement virtuel

git clone https://github.com/<ton-utilisateur>/CREDIT_SCORE_PROJET7.git

cd CREDIT_SCORE_PROJET7

### 2. Créer et activer l’environnement virtuel

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

### 3. Installer les dépendances

pip install -r requirements.txt

---

## 🧬 API FastAPI
### Lancer l’API :

cd api
uvicorn main:app --reload

### Accès :

Interface interactive : http://localhost:8000/docs


---

## 📊 Dashboard Streamlit
### Lancer le dashboard :

cd dashboard
streamlit run app.py


Le dashboard permet :

- d’uploader les fichiers application_test.csv, bureau.csv et previous_application.csv

- de visualiser les prédictions du modèle

- de voir les explications SHAP

---

## ☁️ Déploiement

Ce projet peut être déployé sur :

- Render (API)

- Streamlit Community Cloud (dashboard)


---

## ✅ À venir

- 🔬 Ajout de tests unitaires dans le dossier tests/

- ⚠️ Ajout d’un gestionnaire d’erreur pour les colonnes manquantes

- 📈 Monitoring en production avec MLflow et Evidently

## 🧠 Auteure

Inès Nuckchady
Projet réalisé dans le cadre d’une formation en Data Science