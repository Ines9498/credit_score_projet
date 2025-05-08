import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page Streamlit
st.set_page_config(layout="wide")
st.title("📊 Prédiction de crédit & Explications SHAP")

st.markdown("""
Ce dashboard vous permet d'uploader les fichiers `application_test`, `bureau` et `previous_application`,
de choisir un client (`SK_ID_CURR`), puis d'envoyer les données à une API pour effectuer des prédictions
et visualiser les explications SHAP.
""")

# URL de l'API à contacter (assurez-vous que l'API est bien déployée)
API_URL = "http://0.0.0.0:10000/upload"

# Upload des fichiers CSV
file_app = st.file_uploader("📄 Fichier application_test.csv", type="csv")
file_bureau = st.file_uploader("📄 Fichier bureau.csv", type="csv")
file_prev = st.file_uploader("📄 Fichier previous_application.csv", type="csv")

# Sélection de SK_ID_CURR si le fichier application_test est chargé
sk_id_selected = None
if file_app is not None:
    try:
        df_app_preview = pd.read_csv(file_app, usecols=["SK_ID_CURR"])
        sk_ids = df_app_preview["SK_ID_CURR"].unique().tolist()
        sk_id_selected = st.selectbox("🔎 Choisissez un SK_ID_CURR pour explication locale :", sk_ids)
    except Exception as e:
        st.error(f"❌ Erreur de lecture du fichier application_test.csv : {e}")

# Bouton pour envoyer à l'API
if file_app and file_bureau and file_prev and sk_id_selected is not None:
    if st.button("🚀 Lancer la prédiction"):
        with st.spinner("🧹 Nettoyage des données en cours..."):
            try:
                response = requests.post(
                    API_URL,
                    data={"sk_id_curr": sk_id_selected},
                    files={
                        "application_test": (
                            "application_test.csv", io.BytesIO(file_app.getvalue()), "text/csv"
                        ),
                        "bureau": (
                            "bureau.csv", io.BytesIO(file_bureau.getvalue()), "text/csv"
                        ),
                        "previous_application": (
                            "previous_application.csv", io.BytesIO(file_prev.getvalue()), "text/csv"
                        )
                    }
                )
            except Exception as e:
                st.error(f"❌ Erreur de connexion à l'API : {e}")
                st.stop()

        with st.spinner("🔮 Prédiction et génération des explications SHAP..."):
            if response.status_code == 200:
                try:
                    data = response.json()

                    df_pred = pd.DataFrame(data["predictions"])
                    shap_global_df = pd.DataFrame(data["shap_global"])
                    shap_local_values = data["shap_local"]
                    shap_features = data["features"]

                    st.success("✅ Prédictions reçues avec succès !")

                    # Résultats de prédiction
                    st.subheader("📈 Résultats de la prédiction")
                    st.dataframe(df_pred[df_pred["SK_ID_CURR"] == sk_id_selected])

                    # SHAP global
                    st.subheader("🧠 Importance globale des variables")
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=shap_global_df, x="Importance", y="Feature", ax=ax1)
                    ax1.set_title("Top 20 variables les plus influentes (SHAP global)")
                    st.pyplot(fig1)

                    # SHAP local
                    st.subheader(f"🔍 Explication locale SHAP pour le client {sk_id_selected}")
                    shap_local_df = pd.DataFrame({
                        "Feature": shap_features,
                        "SHAP value": shap_local_values
                    }).sort_values(by="SHAP value", key=np.abs, ascending=False).head(20)

                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=shap_local_df, x="SHAP value", y="Feature", ax=ax2)
                    ax2.set_title(f"Top 20 SHAP locaux pour {sk_id_selected}")
                    st.pyplot(fig2)

                except Exception as e:
                    st.error(f"❌ Erreur lors du traitement de la réponse : {e}")
            else:
                st.error(f"❌ Erreur API ({response.status_code}) : {response.text}")
else:
    st.info("⏳ Veuillez uploader les 3 fichiers et sélectionner un SK_ID_CURR pour activer le bouton.")
