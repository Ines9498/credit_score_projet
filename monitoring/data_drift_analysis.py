# 1. Import des librairies nécessaires
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# 2. Chargement des données
train_path = r"C:\Users\inesn\OneDrive - Université de Paris\credit_score_projet7\data\original\application_train.csv"
test_path = r"C:\Users\inesn\OneDrive - Université de Paris\credit_score_projet7\data\original\application_test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# 3. Suppression de la colonne 'TARGET' pour avoir les mêmes colonnes
if 'TARGET' in df_train.columns:
    df_train = df_train.drop(columns='TARGET')

# 4. Sélection des colonnes communes (évite les erreurs si certaines colonnes sont absentes du test)
common_cols = df_train.columns.intersection(df_test.columns)
df_train_common = df_train[common_cols]
df_test_common = df_test[common_cols]

# 5. Création du rapport Evidently
report = Report(metrics=[DataDriftPreset()])

# 6. Exécution de l'analyse de drift
result = report.run(reference_data=df_train_common, current_data=df_test_common)

# Sauvegarde du rapport HTML
result.save_html("rapport_data_drift.html")
