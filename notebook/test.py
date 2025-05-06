# %%
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# %%
from src.preprocessing import (
    load_all_data,
    explorer_colonnes_interactives,
    analyser_donnees_interactive,
    plot_missing_values,
    supprimer_colonnes_trop_vides,
    imputer_valeurs_manquantes,
    supprimer_lignes_trop_vides,
    convertir_binaires_en_object,
    reduire_types,
    nettoyer_colonnes_categorielles_application,
    nettoyer_colonnes_categorielles_bureau,
    verifier_unicite_id,
    afficher_valeurs_uniques_objet,
    nettoyer_colonnes_categorielles_previous
)

from src.feature_engineering import fusionner_et_agreger_donnees
from src.feature_engineering import feature_engineering_bureau, feature_engineering_previous

# %%
# Chargement des dataset
data_path = r"C:\Users\inesn\OneDrive - Université de Paris\credit_score_projet7\data\original"
data = load_all_data(data_path)

# %%
app_test = data['application_test']

# %%
colonnes_a_conserver = [
    'AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL',
    'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_HOUR',
    'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
    'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_YEAR',
    'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'CODE_GENDER', 'DAYS_BIRTH',
    'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE',
    'DAYS_REGISTRATION', 'DEF_30_CNT_SOCIAL_CIRCLE',
    'DEF_60_CNT_SOCIAL_CIRCLE', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'FLAG_CONT_MOBILE', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
    'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
    'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
    'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_2',
    'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'FLAG_DOCUMENT_3',
    'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
    'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_EMAIL',
    'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'FLAG_PHONE', 'FLAG_WORK_PHONE', 'HOUR_APPR_PROCESS_START',
    'LIVE_CITY_NOT_WORK_CITY', 'LIVE_REGION_NOT_WORK_REGION',
    'NAME_CONTRACT_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE', 'NAME_TYPE_SUITE',
    'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
    'OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'REGION_POPULATION_RELATIVE',
    'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
    'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
    'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
    'SK_ID_CURR', 'WEEKDAY_APPR_PROCESS_START'
]

app_test = app_test[colonnes_a_conserver]

# %%
# Imputation des valeurs manquantes
app_test, app_test_imputations = imputer_valeurs_manquantes(app_test)

# %%
# On convertit certaines colonnes en int car c'est plus pertinent
colonnes_a_convertir_en_int = [
    'CNT_FAM_MEMBERS',
    'OBS_30_CNT_SOCIAL_CIRCLE',
    'DEF_30_CNT_SOCIAL_CIRCLE',
    'OBS_60_CNT_SOCIAL_CIRCLE',
    'DEF_60_CNT_SOCIAL_CIRCLE',
    'AMT_REQ_CREDIT_BUREAU_HOUR',
    'AMT_REQ_CREDIT_BUREAU_DAY',
    'AMT_REQ_CREDIT_BUREAU_WEEK',
    'AMT_REQ_CREDIT_BUREAU_MON',
    'AMT_REQ_CREDIT_BUREAU_QRT',
    'AMT_REQ_CREDIT_BUREAU_YEAR'
]

for col in colonnes_a_convertir_en_int:
    app_test[col] = app_test[col].astype(int)


# %%
# On convertit certaines colonnnes en object car c'est plus pertinent, notamment les flags
app_test, app_test_colonnes_binaires = convertir_binaires_en_object(app_test)

# %%
app_test = nettoyer_colonnes_categorielles_application(app_test)

# %%
# On reduit le type des données pour alléger la mémoire
app_test, app_test_conversions = reduire_types(app_test)

# %%
bureau = data['bureau']

# %%
bureau, bureau_colonnes_supprimees = supprimer_colonnes_trop_vides(bureau, seuil=40)

# %%
bureau, bureau_lignes_supprimees = supprimer_lignes_trop_vides(bureau, seuil=40)

# %%
bureau, bureau_imputations = imputer_valeurs_manquantes(bureau)

# %%
bureau[['DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT']] = bureau[['DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT']].astype('int32')

# %%
bureau = nettoyer_colonnes_categorielles_bureau(bureau)

# %%
bureau, bureau_conversions = reduire_types(bureau)

# %%
previous_application = data['previous_application']

# %%
previous_application, previous_application_colonnes_supprimees = supprimer_colonnes_trop_vides(previous_application, seuil=45)

# %%
previous_application, previous_application_lignes_supprimees = supprimer_lignes_trop_vides(previous_application, seuil=40)

# %%
previous_application, previous_application_imputations = imputer_valeurs_manquantes(previous_application)

# %%
colonnes_a_convertir_int = [
    'CNT_PAYMENT', 'DAYS_DECISION', 'SELLERPLACE_AREA',
    'NFLAG_LAST_APPL_IN_DAY', 'NFLAG_MICRO_CASH', 'NFLAG_INSURED_ON_APPROVAL'
]

for col in colonnes_a_convertir_int:
    if col in previous_application.columns:
        previous_application[col] = previous_application[col].fillna(0).astype(int)


# %%
previous_application, previous_application_colonnes_binaires = convertir_binaires_en_object(previous_application)

# %%
previous_application = nettoyer_colonnes_categorielles_previous(previous_application)

# %%
previous_application, previous_application_conversions = reduire_types(previous_application)

# %%
# Fusion et feature engineering
df = fusionner_et_agreger_donnees(app_test, bureau, previous_application)

# %%
# Remplissage des NaN par 0
df.fillna(0, inplace=True)

# %%
# Nettoyage des noms de colonnes : remplacer les caractères spéciaux et espaces
df.columns = (
    df.columns.str.strip()
              .str.replace('[^A-Za-z0-9_]+', '_', regex=True)
              .str.replace(' ', '_')
)

# Encodage des colonnes catégorielles restantes
cat_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# %%
import joblib

# %%
# Chargement du modèle entraîné (déjà optimisé + seuil choisi)
with open("models/best_model_lightgbm.pkl", "rb") as f:
    best_model = pickle.load(f)

# Chargement du seuil optimal choisi lors de l'entraînement
seuil_optimal = 0.14  # remplace par la bonne valeur si besoin

# Identifiants clients
ids_clients = df["SK_ID_CURR"]

# 1. Retirer SK_ID_CURR
X_app = df.drop(columns=["SK_ID_CURR"])

# 2. Charger les colonnes utilisées à l'entraînement
colonnes_utiles = joblib.load("models/columns_used.pkl")

# 3. Réindexation : ajouter les colonnes manquantes (remplies avec 0) et enlever celles en trop
X_app = X_app.reindex(columns=colonnes_utiles, fill_value=0)

# 4. Charger les types de colonnes (sauvegardés à l'entraînement)
dtypes_dict = joblib.load("models/columns_dtypes.pkl")

# 5. Conversion automatique au bon type
for col, dtype in dtypes_dict.items():
    if col in X_app.columns:
        try:
            X_app[col] = X_app[col].astype(dtype)
        except Exception as e:
            print(f"⚠️ Erreur de conversion pour {col} → {dtype} : {e}")

# Prédiction de la probabilité d'être un "bon" client (classe 1)
probas = best_model.predict_proba(X_app)[:, 1]

# Prédiction finale selon le seuil optimal
predictions = (probas >= seuil_optimal).astype(int)

# Création du DataFrame résultat
resultats = pd.DataFrame({
    "SK_ID_CURR": ids_clients,
    "Score_proba": probas,
    "Decision": predictions
})

# Affichage
print(resultats.head())

# Sauvegarde
resultats.to_csv("resultats_application_test.csv", index=False)



