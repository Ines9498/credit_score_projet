# 🧠 Projet de Scoring Crédit - "Prêt à dépenser"

Ce projet a pour objectif de prédire si un client est un bon ou un mauvais payeur à partir de ses données personnelles, professionnelles et de crédit, en s'appuyant sur les données publiques de Home Credit.  
Il comprend une **API FastAPI** déployée sur Render et un **dashboard Streamlit** permettant de charger les fichiers, faire une prédiction, visualiser des explications SHAP et explorer des indicateurs clés.

---

## 🚀 Fonctionnalités principales

- 🔍 Traitement et agrégation des données (`preprocessing.py`, `feature_engineering.py`)
- 📊 Entraînement d’un modèle **LightGBM** optimisé avec **SMOTE**
- 🧪 Calcul d’un **score métier** pondérant les erreurs critiques
- 🧠 Explicabilité via **SHAP** (globale + locale)
- 🌐 **API FastAPI** pour l'exposition du modèle (✅ déployée sur Render)
- 🖥️ **Dashboard Streamlit** pour :
  - Uploader les fichiers de scoring
  - Sélectionner un client (`SK_ID_CURR`)
  - Obtenir une prédiction "accorder / refuser"
  - Visualiser les valeurs SHAP
  - Comparer les indicateurs clés

- 🧪 **Tests unitaires** pour l’API (`tests/test_api.py`)
- 📉 **Monitoring** de la dérive des données avec **Evidently**

---

## 📁 Structure du projet

credit_score_projet7/
│
├── api/ # Application FastAPI (main.py)
├── dashboard/ # Application Streamlit (app.py)
├── data/ # Données (non suivies par Git)
│ ├── original/ # Jeux de données bruts
│ └── modified/ # Versions transformées
├── models/ # Modèle entraîné + fichiers auxiliaires
│ ├── best_model_lightgbm.pkl
│ ├── columns_used.pkl
│ └── columns_dtypes.pkl
├── notebook/ # Notebook principal
│ └── notebook_credit_score.ipynb
├── monitoring/ # Rapport de data drift Evidently
│ ├── drift_report.html
│ └── venv_drift/ # Environnement spécifique (ignoré)
├── src/ # Prétraitements et feature engineering
│ ├── preprocessing.py
│ └── feature_engineering.py
├── tests/ # Tests automatisés (pytest)
│ └── test_api.py
├── .gitignore
├── requirements.txt
└── README.md

---

## ⚙️ Lancer le projet localement

git clone https://github.com/<ton-utilisateur>/credit_score_projet7.git
cd credit_score_projet7

python -m venv venv
source venv/bin/activate        # (ou venv\Scripts\activate sous Windows)

pip install -r requirements.txt

🧬 API FastAPI
Lancer l’API localement 

cd api
uvicorn main:app --reload

Accès local :

  -  Interface interactive : http://localhost:8000/docs

Accès en ligne :

    ✅ API déployée sur Render

📊 Dashboard Streamlit
Lancer localement :

cd dashboard
streamlit run app.py

Le dashboard permet :

  - de charger les fichiers application_test.csv, bureau.csv, previous_application.csv

  - de sélectionner un SK_ID_CURR

  - d'obtenir la prédiction et les explications SHAP

🧪 Tests & Monitoring
✅ Lancer les tests unitaires :

pytest tests/test_api.py

📈 Rapport de dérive des données :

Un rapport Evidently a été généré pour comparer application_train.csv (référence) et application_test.csv (production) :

    📄 monitoring/drift_report.html

☁️ Déploiement

  - API FastAPI : https://api-credit-score.onrender.com

  - Dashboard Streamlit : déploiement possible sur Streamlit Cloud

✅ Améliorations prévues

- Ajouter des tests pour les erreurs

- Mettre en place l’analyse de dérive des données

- Ajouter l’authentification sur l’API

- Affiner le seuil de décision métier dynamiquement

🧠 Auteure

Inès Nuckchady
Projet réalisé dans le cadre d’une formation en Data Science.