# credit_scoring_api.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import numpy as np
import uvicorn
import shap
import pickle

from io import BytesIO
from src.preprocessing import (
    imputer_valeurs_manquantes,
    convertir_binaires_en_object,
    reduire_types,
    nettoyer_colonnes_categorielles_application,
    nettoyer_colonnes_categorielles_bureau,
    nettoyer_colonnes_categorielles_previous,
)
from src.feature_engineering import fusionner_et_agreger_donnees

app = FastAPI()

# Chargement des objets en mémoire
model = pickle.load(open("models/best_model_lightgbm.pkl", "rb"))
colonnes_utiles = joblib.load("models/columns_used.pkl")
dtypes_dict = joblib.load("models/columns_dtypes.pkl")
seuil_optimal = 0.14

@app.post("/predict")
async def predict(application_test: UploadFile = File(...),
                  bureau: UploadFile = File(...),
                  previous_application: UploadFile = File(...),
                  sk_id: int = None):
    try:
        # Chargement des fichiers CSV
        app_test = pd.read_csv(BytesIO(await application_test.read()))
        bureau_df = pd.read_csv(BytesIO(await bureau.read()))
        previous_df = pd.read_csv(BytesIO(await previous_application.read()))

        # Colonnes à conserver dans app_test
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

        # Prétraitement app_test
        app_test, _ = imputer_valeurs_manquantes(app_test)
        app_test, _ = convertir_binaires_en_object(app_test)
        app_test = nettoyer_colonnes_categorielles_application(app_test)
        app_test, _ = reduire_types(app_test)

        bureau_df, _ = nettoyer_colonnes_categorielles_bureau(bureau_df)
        bureau_df, _ = reduire_types(bureau_df)
        previous_df, _ = nettoyer_colonnes_categorielles_previous(previous_df)
        previous_df, _ = reduire_types(previous_df)

        df = fusionner_et_agreger_donnees(app_test, bureau_df, previous_df)
        df.fillna(0, inplace=True)
        df.columns = df.columns.str.strip().str.replace('[^A-Za-z0-9_]+', '_', regex=True)

        cat_cols = df.select_dtypes(include='object').columns
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        ids_clients = df['SK_ID_CURR']
        X_app = df.drop(columns=['SK_ID_CURR'])
        X_app = X_app.reindex(columns=colonnes_utiles, fill_value=0)

        for col, dtype in dtypes_dict.items():
            if col in X_app.columns:
                try:
                    X_app[col] = X_app[col].astype(dtype)
                except Exception as e:
                    print(f"Erreur sur {col} : {e}")

        probas = model.predict_proba(X_app)[:, 1]
        preds = (probas >= seuil_optimal).astype(int)

        df_results = pd.DataFrame({
            "SK_ID_CURR": ids_clients,
            "Score_proba": probas,
            "Decision": preds
        })

        if sk_id is not None:
            if sk_id not in ids_clients.values:
                raise HTTPException(status_code=404, detail="SK_ID_CURR non trouvé")
            index = ids_clients[ids_clients == sk_id].index[0]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_app)
            valeurs = shap_values[1][index]
            features = X_app.iloc[index]
            shap_data = dict(sorted(zip(features.index, valeurs), key=lambda x: abs(x[1]), reverse=True)[:10])
            return JSONResponse({
                "prediction": int(preds[index]),
                "proba": float(probas[index]),
                "shap_top_10": shap_data
            })

        return df_results.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
