import streamlit as st
import pandas as pd
import requests
import io
import base64
from PIL import Image

st.set_page_config(layout="wide")
st.title("\ud83d\udcca Pr\u00e9diction de cr\u00e9dit & SHAP")

API_URL = "https://api-credit-score.onrender.com/upload"

file_app = st.file_uploader("\ud83d\udcc4 Fichier application_test.csv", type="csv")
file_bureau = st.file_uploader("\ud83d\udcc4 Fichier bureau.csv", type="csv")
file_prev = st.file_uploader("\ud83d\udcc4 Fichier previous_application.csv", type="csv")

sk_id_selected = None
if file_app is not None:
    try:
        file_app.seek(0)
        df_app_preview = pd.read_csv(file_app, usecols=["SK_ID_CURR"])
        sk_ids = df_app_preview["SK_ID_CURR"].unique().tolist()
        sk_id_selected = st.selectbox("\ud83d\udd0e Choisissez un SK_ID_CURR :", sk_ids)
    except Exception as e:
        st.error(f"\u274c Erreur fichier application_test.csv : {e}")

if file_app and file_bureau and file_prev and sk_id_selected is not None:
    if st.button("\ud83d\ude80 Lancer la pr\u00e9diction"):
        file_app.seek(0)
        file_bureau.seek(0)
        file_prev.seek(0)

        with st.spinner("\ud83e\udde0 Pr\u00e9diction en cours..."):
            response = requests.post(
                API_URL,
                data={"sk_id_curr": sk_id_selected},
                files={
                    "application_test": ("application_test.csv", file_app, "text/csv"),
                    "bureau": ("bureau.csv", file_bureau, "text/csv"),
                    "previous_application": ("previous_application.csv", file_prev, "text/csv")
                }
            )

        if response.status_code == 200:
            data = response.json()
            df_pred = pd.DataFrame(data["predictions"])
            st.subheader("\ud83d\udcc8 R\u00e9sultat de la pr\u00e9diction")
            st.dataframe(df_pred[df_pred["SK_ID_CURR"] == sk_id_selected])

            # Infos contextuelles du client
            if "infos_contextuelles" in data:
                st.subheader("\ud83e\uddd0 Informations du client")
                client_info = data["infos_contextuelles"]
                st.markdown(f"""
                - **\u00c2ge** : {client_info['Age_annees']} ans  
                - **Revenu total** : {client_info['AMT_INCOME_TOTAL']:.0f}  
                - **Montant du cr\u00e9dit** : {client_info['AMT_CREDIT']:.0f}  
                - **Statut familial** : {client_info['NAME_FAMILY_STATUS']}  
                - **Type de logement** : {client_info['NAME_HOUSING_TYPE']}  
                - **Profession** : {client_info['OCCUPATION_TYPE']}
                """)

            # Moyennes des autres clients
            if "comparaison_moyenne" in data:
                st.subheader("\ud83d\udcca Comparaison avec la moyenne des clients")
                moyenne_info = data["comparaison_moyenne"]
                st.markdown(f"""
                - **\u00c2ge moyen** : {moyenne_info['Age_annees']} ans  
                - **Revenu moyen** : {moyenne_info['AMT_INCOME_TOTAL']:.0f}  
                - **Montant de cr\u00e9dit moyen** : {moyenne_info['AMT_CREDIT']:.0f}
                """)

            # SHAP Summary Plot
            if data.get("shap_summary_plot"):
                st.subheader("\ud83d\udcc9 SHAP Summary Plot (Global)")
                summary_img = Image.open(io.BytesIO(base64.b64decode(data["shap_summary_plot"])))
                st.image(summary_img, caption="Summary Plot des SHAP values")
            else:
                st.warning("\u26a0\ufe0f Le graphique SHAP global n'a pas pu \u00eatre g\u00e9n\u00e9r\u00e9.")

            # SHAP Force Plot
            if data.get("shap_force_plot"):
                st.subheader("\u26a1 SHAP Force Plot (Client sp\u00e9cifique)")
                force_img = Image.open(io.BytesIO(base64.b64decode(data["shap_force_plot"])))
                st.image(force_img, caption=f"Force Plot pour SK_ID_CURR {sk_id_selected}")
            else:
                st.warning("\u26a0\ufe0f Le graphique SHAP local n'a pas pu \u00eatre g\u00e9n\u00e9r\u00e9.")

        else:
            st.error(f"\u274c Erreur API ({response.status_code}) : {response.text}")
else:
    st.info("\u23f3 Uploadez les fichiers et s\u00e9lectionnez un SK_ID_CURR pour activer la pr\u00e9diction.")
