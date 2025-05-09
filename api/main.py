# === FICHIER API (FastAPI) COMPLET ===
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import base64
import os
import io
import joblib
import pickle

from src.preprocessing import (
    imputer_valeurs_manquantes,
    convertir_binaires_en_object,
    reduire_types,
    nettoyer_colonnes_categorielles_application,
    nettoyer_colonnes_categorielles_bureau,
    nettoyer_colonnes_categorielles_previous
)
from src.feature_engineering import fusionner_et_agreger_donnees

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement initial des modèles
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
with open(os.path.join(base_dir, "best_model_lightgbm.pkl"), "rb") as f:
    model = pickle.load(f)
colonnes_utiles = joblib.load(os.path.join(base_dir, "columns_used.pkl"))
colonnes_types = joblib.load(os.path.join(base_dir, "columns_dtypes.pkl"))
explainer = shap.TreeExplainer(model)

@app.post("/upload")
async def upload_files(
    application_test: UploadFile = File(...),
    bureau: UploadFile = File(...),
    previous_application: UploadFile = File(...),
    sk_id_curr: int = Form(...)
):
    try:
        df_app = pd.read_csv(io.BytesIO(await application_test.read()))
        df_bureau = pd.read_csv(io.BytesIO(await bureau.read()))
        df_prev = pd.read_csv(io.BytesIO(await previous_application.read()))

        # === Prétraitement application_test ===
        app_colonnes_a_conserver = [
            'AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL',
            'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON',
            'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_YEAR',
            'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'CODE_GENDER', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
            'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_REGISTRATION',
            'DEF_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
            'FLAG_CONT_MOBILE', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
            'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
            'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_2',
            'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
            'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
            'FLAG_DOCUMENT_9', 'FLAG_EMAIL', 'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_OWN_CAR',
            'FLAG_OWN_REALTY', 'FLAG_PHONE', 'FLAG_WORK_PHONE', 'HOUR_APPR_PROCESS_START',
            'LIVE_CITY_NOT_WORK_CITY', 'LIVE_REGION_NOT_WORK_REGION', 'NAME_CONTRACT_TYPE',
            'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE',
            'NAME_TYPE_SUITE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
            'OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'REGION_POPULATION_RELATIVE',
            'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'REG_CITY_NOT_LIVE_CITY',
            'REG_CITY_NOT_WORK_CITY', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
            'SK_ID_CURR', 'WEEKDAY_APPR_PROCESS_START'
        ]
        df_app = df_app[app_colonnes_a_conserver]
        df_app, _ = imputer_valeurs_manquantes(df_app)

        for col in ['CNT_FAM_MEMBERS', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
                    'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
                    'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
                    'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
                    'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']:
            df_app[col] = df_app[col].astype(int)

        df_app, _ = convertir_binaires_en_object(df_app)
        df_app = nettoyer_colonnes_categorielles_application(df_app)
        df_app, _ = reduire_types(df_app)

        # === Prétraitement bureau ===
        bureau_colonnes_a_conserver = [
            'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
            'AMT_CREDIT_SUM_OVERDUE', 'CNT_CREDIT_PROLONG', 'CREDIT_ACTIVE',
            'CREDIT_CURRENCY', 'CREDIT_DAY_OVERDUE', 'CREDIT_TYPE', 'DAYS_CREDIT',
            'DAYS_CREDIT_ENDDATE', 'DAYS_CREDIT_UPDATE', 'DAYS_ENDDATE_FACT',
            'SK_ID_BUREAU', 'SK_ID_CURR'
        ]
        df_bureau = df_bureau[bureau_colonnes_a_conserver]
        df_bureau, _ = imputer_valeurs_manquantes(df_bureau)
        df_bureau[['DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT']] = df_bureau[
            ['DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT']
        ].astype('int32')
        df_bureau = nettoyer_colonnes_categorielles_bureau(df_bureau)
        df_bureau, _ = reduire_types(df_bureau)

        # === Prétraitement previous_application ===
        prev_colonnes_a_conserver = [
            'AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_GOODS_PRICE',
            'CHANNEL_TYPE','CNT_PAYMENT','CODE_REJECT_REASON','DAYS_DECISION',
            'DAYS_FIRST_DRAWING','DAYS_FIRST_DUE','DAYS_LAST_DUE','DAYS_LAST_DUE_1ST_VERSION',
            'DAYS_TERMINATION','FLAG_LAST_APPL_PER_CONTRACT','HOUR_APPR_PROCESS_START',
            'NAME_CASH_LOAN_PURPOSE','NAME_CLIENT_TYPE','NAME_CONTRACT_STATUS',
            'NAME_CONTRACT_TYPE','NAME_GOODS_CATEGORY','NAME_PAYMENT_TYPE',
            'NAME_PORTFOLIO','NAME_PRODUCT_TYPE','NAME_SELLER_INDUSTRY',
            'NAME_YIELD_GROUP','NFLAG_INSURED_ON_APPROVAL','NFLAG_LAST_APPL_IN_DAY',
            'PRODUCT_COMBINATION','SELLERPLACE_AREA','SK_ID_CURR','SK_ID_PREV',
            'WEEKDAY_APPR_PROCESS_START'
        ]
        df_prev = df_prev[prev_colonnes_a_conserver]
        df_prev, _ = imputer_valeurs_manquantes(df_prev)

        for col in ['CNT_PAYMENT', 'DAYS_DECISION', 'SELLERPLACE_AREA',
                    'NFLAG_LAST_APPL_IN_DAY', 'NFLAG_MICRO_CASH', 'NFLAG_INSURED_ON_APPROVAL']:
            if col in df_prev.columns:
                df_prev[col] = df_prev[col].fillna(0).astype(int)

        df_prev, _ = convertir_binaires_en_object(df_prev)
        df_prev = nettoyer_colonnes_categorielles_previous(df_prev)
        df_prev, _ = reduire_types(df_prev)

        # === Fusion & Feature engineering ===
        df = fusionner_et_agreger_donnees(df_app, df_bureau, df_prev)
        df.fillna(0, inplace=True)
        df.columns = df.columns.str.strip().str.replace('[^A-Za-z0-9_]+', '_', regex=True)
        df = pd.get_dummies(df, columns=df.select_dtypes(include='object').columns, drop_first=True)

        ids_clients = df["SK_ID_CURR"]
        X = df.drop(columns=["SK_ID_CURR"]).reindex(columns=colonnes_utiles, fill_value=0)
        for col, dtype in colonnes_types.items():
            if col in X.columns:
                X[col] = X[col].astype(dtype)

        probas = model.predict_proba(X)[:, 1]
        seuil = 0.14
        y_pred = (probas >= seuil).astype(int)

        resultats = pd.DataFrame({
            "SK_ID_CURR": ids_clients,
            "Score_proba": probas,
            "Decision": y_pred
        })

        shap_vals = explainer.shap_values(X)
        idx = ids_clients[ids_clients == sk_id_curr].index[0]

        shap_values_summary = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

        try:
            fig_summary, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values_summary, X, show=False)
            buf_summary = io.BytesIO()
            plt.savefig(buf_summary, format="png", bbox_inches="tight")
            plt.close(fig_summary)
            summary_plot_b64 = base64.b64encode(buf_summary.getvalue()).decode("utf-8")
        except Exception:
            summary_plot_b64 = None

        try:
            shap_values_force = shap_vals[1][idx] if isinstance(shap_vals, list) else shap_vals[idx]
            expected_value_force = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            fig_force = plt.figure()
            shap.force_plot(
                expected_value_force, shap_values_force, X.iloc[idx], matplotlib=True, show=False
            )
            buf_force = io.BytesIO()
            plt.savefig(buf_force, format="png", bbox_inches="tight")
            plt.close(fig_force)
            force_plot_b64 = base64.b64encode(buf_force.getvalue()).decode("utf-8")
        except Exception:
            force_plot_b64 = None

        return JSONResponse(content={
            "predictions": resultats.to_dict(orient="records"),
            "shap_summary_plot": summary_plot_b64,
            "shap_force_plot": force_plot_b64
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def home():
    return {"message": "API de scoring crédit opérationnelle 🚀 - accédez à /docs pour voir les endpoints."}