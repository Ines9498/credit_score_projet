from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from io import BytesIO
import joblib
import pickle
import shap
import os
import logging

# Initialisation logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "API opérationnelle. Accédez à /docs pour tester."}

@app.post("/upload")
async def upload_files(
    application_test: UploadFile = File(...),
    bureau: UploadFile = File(...),
    previous_application: UploadFile = File(...),
    sk_id_curr: int = Form(...)
):
    try:
        logger.info("📥 Lecture des fichiers")
        df_app = pd.read_csv(BytesIO(await application_test.read()))
        df_bureau = pd.read_csv(BytesIO(await bureau.read()))
        df_prev = pd.read_csv(BytesIO(await previous_application.read()))

        logger.info("📊 Limitation à 1000 lignes pour debug")
        df_app = df_app.head(1000)
        df_bureau = df_bureau[df_bureau["SK_ID_CURR"].isin(df_app["SK_ID_CURR"])]
        df_prev = df_prev[df_prev["SK_ID_CURR"].isin(df_app["SK_ID_CURR"])]

        # 🧹 Simule prétraitement rapide (remplace tes vraies fonctions si besoin)
        df_app.fillna(0, inplace=True)
        df_bureau.fillna(0, inplace=True)
        df_prev.fillna(0, inplace=True)

        logger.info("🔗 Fusion des données")
        df = df_app.merge(df_bureau.groupby("SK_ID_CURR").mean(), on="SK_ID_CURR", how="left")
        df = df.merge(df_prev.groupby("SK_ID_CURR").mean(), on="SK_ID_CURR", how="left")
        df.fillna(0, inplace=True)

        logger.info("🧠 Chargement du modèle")
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        model = pickle.load(open(os.path.join(base_dir, "best_model_lightgbm.pkl"), "rb"))
        colonnes_utiles = joblib.load(os.path.join(base_dir, "columns_used.pkl"))
        colonnes_types = joblib.load(os.path.join(base_dir, "columns_dtypes.pkl"))

        logger.info("🎯 Préparation des features")
        ids_clients = df["SK_ID_CURR"]
        X = df.drop(columns=["SK_ID_CURR"]).reindex(columns=colonnes_utiles, fill_value=0)
        for col, dtype in colonnes_types.items():
            if col in X.columns:
                X[col] = X[col].astype(dtype)

        logger.info("🔮 Prédiction")
        probas = model.predict_proba(X)[:, 1]
        y_pred = (probas >= 0.14).astype(int)

        resultats = pd.DataFrame({
            "SK_ID_CURR": ids_clients,
            "Score_proba": probas,
            "Decision": y_pred
        })

        if sk_id_curr not in ids_clients.values:
            raise HTTPException(status_code=404, detail=f"SK_ID_CURR {sk_id_curr} introuvable.")

        logger.info("📈 Calcul SHAP global et local")
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X.sample(100))  # global SHAP sur 100 exemples

        idx = ids_clients[ids_clients == sk_id_curr].index[0]
        shap_local = explainer.shap_values(X)[1][idx].tolist()
        shap_global = np.abs(shap_vals[1]).mean(axis=0)
        shap_global_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": shap_global
        }).sort_values(by="Importance", ascending=False).head(20)

        logger.info("✅ Réponse prête")
        return JSONResponse(content={
            "predictions": resultats.to_dict(orient="records"),
            "shap_global": shap_global_df.to_dict(orient="records"),
            "shap_local": shap_local,
            "features": X.columns.tolist(),
            "ids_clients": ids_clients.tolist()
        })

    except Exception as e:
        logger.error(f"❌ Erreur dans /upload : {e}")
        raise HTTPException(status_code=500, detail=str(e))
