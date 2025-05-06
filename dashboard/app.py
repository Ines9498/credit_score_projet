import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("📊 Dashboard de Scoring Crédit")

st.markdown("""
Ce dashboard vous permet d'uploader les fichiers de test et d'obtenir une prédiction de crédit
pour un client sélectionné, accompagnée d'une explication SHAP.
""")

# === Upload des fichiers ===
with st.sidebar:
    st.header("1. Upload des fichiers")
    file_app = st.file_uploader("Fichier application_test.csv", type="csv")
    file_bureau = st.file_uploader("Fichier bureau.csv", type="csv")
    file_prev = st.file_uploader("Fichier previous_application.csv", type="csv")

# === Étape 2 : Choix du client ===
if file_app and file_bureau and file_prev:
    df_app = pd.read_csv(file_app)
    ids = df_app["SK_ID_CURR"]
    selected_id = st.selectbox("2. Choisir un client à prédire :", ids)

    if st.button("Obtenir la prédiction"):
        with st.spinner("🔍 Prédiction en cours..."):
            response = requests.post(
                "https://nom_de_ton_api_render.onrender.com/predict",  # À modifier après déploiement
                files={
                    "application_test": ("app.csv", file_app),
                    "bureau": ("bureau.csv", file_bureau),
                    "previous_application": ("prev.csv", file_prev)
                },
                data={"sk_id_curr": selected_id}
            )
            if response.status_code == 200:
                result = response.json()

                st.success("✅ Prédiction reçue !")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probabilité d'accord", f"{result['Score_proba']:.2f}")
                with col2:
                    decision = "✅ Accorder crédit" if result['Decision'] else "❌ Refuser crédit"
                    st.metric("Décision", decision)

                if result["shap_local"]:
                    shap_df = pd.Series(result["shap_local"]).sort_values(key=abs, ascending=False).head(10)
                    st.subheader("🔍 SHAP local - Impact des variables")
                    fig, ax = plt.subplots()
                    shap_df.plot(kind="barh", ax=ax, color=["green" if v > 0 else "red" for v in shap_df])
                    ax.set_title("Top 10 des variables les plus impactantes")
                    st.pyplot(fig)
            else:
                st.error(f"❌ Erreur de l'API : {response.text}")
else:
    st.info("Veuillez uploader les trois fichiers pour commencer.")
