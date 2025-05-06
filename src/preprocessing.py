# =============================================================================
# 📁 IMPORTS
# =============================================================================

import os
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

# =============================================================================
# 📂 CHARGEMENT DES FICHIERS
# =============================================================================

def load_all_data(directory_path):
    """
    Charge tous les fichiers CSV du dossier dans un dictionnaire.
    Essaie plusieurs encodages en cas d'erreur.
    """
    encodings_to_try = ['utf-8', 'iso-8859-1', 'cp1252', 'utf-16']
    data = {}

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            name = filename.replace(".csv", "")
            path = os.path.join(directory_path, filename)
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(path, encoding=encoding)
                    data[name] = df
                    print(f"✅ Fichier {filename} chargé avec encodage {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print(f"❌ Impossible de lire {filename} avec les encodages testés.")
    return data

# =============================================================================
# 🔍 EXPLORATION DES DONNÉES
# =============================================================================

# Exploration des colonnes 

def explorer_colonnes_interactives(df, n=10, table=None):
    """
    Affiche un aperçu interactif d’un DataFrame contenant des colonnes et leur description.

    Parameters:
    - df : DataFrame contenant au moins les colonnes 'Row', 'Description' et éventuellement 'Table'
    - n : nombre de lignes à afficher par page
    - table : nom de la table à filtrer (ex: 'application', 'bureau', etc.)
    """
    # Vérifie que les colonnes de base sont bien présentes
    required_columns = {'Row', 'Description'}
    if not required_columns.issubset(df.columns):
        print("❌ Le DataFrame doit contenir au minimum les colonnes 'Row' et 'Description'")
        return

    desc = df.copy()

    # Filtrage éventuel par table
    if table and 'Table' in desc.columns:
        desc = desc[desc['Table'].str.lower() == table.lower()]
        if desc.empty:
            print(f"⚠️ Aucune colonne trouvée pour la table '{table}'")
            return

    desc = desc[['Row', 'Description']].reset_index(drop=True)

    def show_page(page=0):
        start = page * n
        end = start + n
        display(desc.iloc[start:end])

    max_page = max(0, (len(desc) - 1) // n)
    page_slider = widgets.IntSlider(min=0, max=max_page, step=1, description='Page')
    widgets.interact(show_page, page=page_slider)

# Analyse des données

def analyser_donnees_interactive(df, n=10):
    """
    Affiche une analyse interactive d’un DataFrame :
    types, valeurs manquantes, nombre de valeurs uniques, remplissage.
    
    Affiche par page n lignes via un slider.
    """
    summary = pd.DataFrame({
        'Column': df.columns,
        'Dtype': df.dtypes.values,
        'Missing(%)': df.isnull().mean().values * 100,
        'Unique': df.nunique().values
    })

    summary['Remplissage(%)'] = 100 - summary['Missing(%)']
    summary = summary.sort_values(by='Remplissage(%)', ascending=False).reset_index(drop=True)

    def show_page(page=0):
        start = page * n
        end = start + n
        display(summary.iloc[start:end])

    max_page = max(0, (len(summary) - 1) // n)
    page_slider = widgets.IntSlider(min=0, max=max_page, step=1, description='Page')
    widgets.interact(show_page, page=page_slider)

# Regarder si les id sont uniques 

def verifier_unicite_id(df, id_col='SK_ID_CURR', nom_df='DataFrame'):
    """
    Vérifie si chaque identifiant dans `id_col` est unique dans le DataFrame.
    
    Paramètres :
    - df : DataFrame à analyser
    - id_col : nom de la colonne identifiant à vérifier (par défaut 'SK_ID_CURR')
    - nom_df : nom optionnel du dataset pour l'affichage

    Affiche un message clair et retourne True si unique, sinon False.
    """
    nb_lignes = df.shape[0]
    nb_ids_uniques = df[id_col].nunique()

    print(f"📊 Vérification pour {nom_df}")
    print(f"Nombre de lignes          : {nb_lignes}")
    print(f"Nombre d'identifiants uniques : {nb_ids_uniques}")

    if nb_lignes == nb_ids_uniques:
        print("✅ Il y a UNE ligne par client (identifiant unique).")
        return True
    else:
        print("⚠️ Il y a PLUSIEURS lignes par client (identifiant non unique).")
        return False

# =============================================================================
# 📉 ANALYSE DES VALEURS MANQUANTES
# =============================================================================

# Valeurs manquantes 

def plot_missing_values(df, figsize=(10, 8)):
    """
    Affiche un graphique des colonnes avec des valeurs manquantes,
    triées de la plus vide à la moins vide, avec couleur selon seuils.
    """
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("✅ Aucune valeur manquante détectée.")
        return

    # Couleur par seuil
    colors = []
    for val in missing:
        if val > 75:
            colors.append('#d73027')  # rouge foncé
        elif val > 50:
            colors.append('#fc8d59')  # orange
        elif val > 25:
            colors.append('#fee08b')  # jaune
        else:
            colors.append('#d9ef8b')  # vert clair

    # Affichage
    plt.figure(figsize=figsize)
    bars = plt.barh(missing.index, missing.values, color=colors)
    plt.xlabel("Taux de valeurs manquantes (%)")
    plt.title("Colonnes avec valeurs manquantes")
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
# Supprimer colonnes 

def supprimer_colonnes_trop_vides(df, seuil=40):
    """
    Supprime les colonnes dont plus de `seuil`% des valeurs sont manquantes.
    
    Retourne le DataFrame nettoyé + liste des colonnes supprimées.
    """
    missing = df.isnull().mean() * 100
    cols_to_drop = missing[missing > seuil].index.tolist()
    
    print(f"🧹 Colonnes supprimées (>{seuil}% de valeurs manquantes) :")
    for col in cols_to_drop:
        print(f"  - {col}")
    
    df_cleaned = df.drop(columns=cols_to_drop)
    return df_cleaned, cols_to_drop

# Supprimer ligne avec trop de valeurs manquantes

def supprimer_lignes_trop_vides(df, seuil=40):
    """
    Supprime les lignes contenant plus de `seuil`% de valeurs manquantes.

    Retourne :
    - le DataFrame nettoyé
    - la liste des index supprimés
    """
    pourcentage_manquant = df.isnull().mean(axis=1) * 100
    lignes_a_supprimer = df.index[pourcentage_manquant > seuil].tolist()

    print(f"🧹 Lignes supprimées (>{seuil}% de colonnes manquantes) : {len(lignes_a_supprimer)} lignes")
    
    df_cleaned = df.drop(index=lignes_a_supprimer)
    return df_cleaned, lignes_a_supprimer

# =============================================================================
# 🧩 IMPUTATION
# =============================================================================

# Imputation des valeurs manquantes

def imputer_valeurs_manquantes(df):
    """
    Impute les valeurs manquantes :
    - Moyenne pour float
    - Moyenne arrondie vers le bas pour int
    - Valeur la plus fréquente (mode) pour les objets (catégories)
    
    Retourne le DataFrame imputé + un dictionnaire des valeurs utilisées.
    """
    df = df.copy()
    imputations = {}

    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'float64':
                valeur = df[col].mean()
                df[col].fillna(valeur, inplace=True)
                imputations[col] = f"float → moyenne = {valeur:.2f}"

            elif df[col].dtype == 'int64':
                moyenne = df[col].mean()
                valeur = int(np.floor(moyenne))
                df[col].fillna(valeur, inplace=True)
                imputations[col] = f"int → moyenne arrondie = {valeur}"

            elif df[col].dtype == 'object':
                valeur = df[col].mode()[0]
                df[col].fillna(valeur, inplace=True)
                imputations[col] = f"object → mode = {valeur}"

            else:
                imputations[col] = "⚠️ Type non géré, aucune imputation"
    
    print("🧩 Imputation des valeurs manquantes effectuée :")
    for col, val in imputations.items():
        print(f"  - {col} : {val}")
    
    return df, imputations

# =============================================================================
# 🧠 TYPE & FORMAT DES DONNÉES
# =============================================================================

# Conversion des binaires

def convertir_binaires_en_object(df, exclude=['TARGET']):
    """
    Convertit en type 'object' toutes les colonnes numériques (int ou float)
    contenant 1 ou 2 valeurs uniques (hors NaN), typiquement 0 et 1 ou constantes,
    sauf celles indiquées dans `exclude`.

    Retourne le DataFrame modifié + la liste des colonnes converties.
    """
    df = df.copy()
    colonnes_converties = []

    for col in df.columns:
        if col in exclude:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            uniques = df[col].dropna().unique()
            if len(uniques) <= 2:
                df[col] = df[col].astype('object')
                colonnes_converties.append(col)

    print(f"🔁 Colonnes binaires ou constantes converties en 'object' (hors {exclude}) :")
    print(colonnes_converties)
    return df, colonnes_converties


# Réduire les types

def reduire_types(df):
    """
    Réduit les types des colonnes numériques si possible :
    - int64 → int8 / int16 selon les valeurs
    - float64 → float32
    Ne modifie pas les colonnes de type object ou category.
    
    Retourne le DataFrame modifié + un résumé des conversions.
    """
    df = df.copy()
    conversions = []

    for col in df.select_dtypes(include=['int64']).columns:
        min_val, max_val = df[col].min(), df[col].max()
        if min_val >= -128 and max_val <= 127:
            df[col] = df[col].astype('int8')
            conversions.append((col, 'int64', 'int8'))
        elif min_val >= -32768 and max_val <= 32767:
            df[col] = df[col].astype('int16')
            conversions.append((col, 'int64', 'int16'))
        else:
            df[col] = df[col].astype('int32')
            conversions.append((col, 'int64', 'int32'))

    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
        conversions.append((col, 'float64', 'float32'))

    print("📉 Colonnes converties (types numériques uniquement) :")
    for col, old, new in conversions:
        print(f"  - {col} : {old} → {new}")
    
    return df, conversions

# =============================================================================
# 🧹 NETTOYAGE DES VALEURS CATÉGORIELLES
# =============================================================================

# Afficher les valeurs uniques des types object qui contiennent plus de 2 valeurs uniques 

def afficher_valeurs_uniques_objet(df, seuil=2, nom_df='DataFrame'):
    """
    Affiche les colonnes de type object qui contiennent plus de `seuil` valeurs uniques (hors NaN),
    ainsi que la fréquence de chaque modalité.

    Paramètres :
    - df : DataFrame à analyser
    - seuil : nombre minimal de valeurs uniques pour affichage (défaut = 2)
    - nom_df : nom du dataset pour l'affichage (optionnel)
    """
    print(f"🔍 Analyse des colonnes 'object' de {nom_df} avec > {seuil} valeurs uniques :\n")
    
    obj_cols = df.select_dtypes(include='object')
    obj_cols_multi = [col for col in obj_cols.columns if df[col].nunique(dropna=True) > seuil]

    if not obj_cols_multi:
        print("✅ Aucune colonne object avec plus de", seuil, "valeurs uniques.")
        return

    for col in obj_cols_multi:
        print(f"🔸 {col} ({df[col].nunique(dropna=True)} valeurs uniques) :")
        print(df[col].value_counts(dropna=False))
        print("-" * 60)

# Regroupement des valeurs 

# Application

def regrouper_organisation(org):
    if pd.isna(org):
        return np.nan
    elif org.startswith('Business Entity'):
        return 'Business'
    elif org.startswith('Trade'):
        return 'Trade'
    elif org.startswith('Industry'):
        return 'Industry'
    elif org.startswith('Transport'):
        return 'Transport'
    elif org in ['Government', 'Police', 'Security Ministries']:
        return 'Government'
    elif org in ['Medicine', 'Emergency', 'Kindergarten']:
        return 'Health'
    elif org in ['Restaurant', 'Services', 'Cleaning', 'Postal', 'Telecom', 'Mobile']:
        return 'Services'
    elif org in ['School', 'University']:
        return 'Education'
    else:
        return 'Other'

def nettoyer_colonnes_categorielles_application(df):
    """
    Nettoie et regroupe les colonnes catégorielles de application_train.csv pour réduire la cardinalité
    et supprimer les modalités très rares ou peu interprétables.
    """
    df = df.copy()

    # Regroupement de NAME_TYPE_SUITE
    df['NAME_TYPE_SUITE'] = df['NAME_TYPE_SUITE'].replace({
        'Spouse, partner': 'Family',
        'Family': 'Family',
        'Children': 'Family',
        'Other_A': 'Other',
        'Other_B': 'Other',
        'Group of people': 'Other'
    })

    # Regroupement des modalités rares dans NAME_INCOME_TYPE
    rares = ['Unemployed', 'Student', 'Businessman', 'Maternity leave']
    df['NAME_INCOME_TYPE'] = df['NAME_INCOME_TYPE'].replace(rares, 'Other')

    # Regroupement de NAME_EDUCATION_TYPE
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].replace({
        'Secondary / secondary special': 'Secondary',
        'Lower secondary': 'Secondary',
        'Incomplete higher': 'Some college',
        'Higher education': 'Higher',
        'Academic degree': 'Higher'
    })

    # Suppression des lignes avec valeurs incohérentes
    df = df[df['CODE_GENDER'] != 'XNA']
    df = df[df['NAME_FAMILY_STATUS'] != 'Unknown']

    # Regroupement de NAME_HOUSING_TYPE
    df['NAME_HOUSING_TYPE'] = df['NAME_HOUSING_TYPE'].replace({
        'With parents': 'Other',
        'Municipal apartment': 'Other',
        'Rented apartment': 'Other',
        'Office apartment': 'Other',
        'Co-op apartment': 'Other'
    })

    # Regroupement de OCCUPATION_TYPE
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].replace({
        'Laborers': 'Labor',
        'Drivers': 'Labor',
        'Low-skill Laborers': 'Labor',
        'Cleaning staff': 'Labor',
        'Sales staff': 'Service',
        'Security staff': 'Service',
        'Cooking staff': 'Service',
        'Waiters/barmen staff': 'Service',
        'Private service staff': 'Service',
        'High skill tech staff': 'Technical',
        'IT staff': 'Technical',
        'Accountants': 'Technical',
        'Medicine staff': 'Technical',
        'Core staff': 'Administrative',
        'Managers': 'Administrative',
        'HR staff': 'Administrative',
        'Secretaries': 'Administrative',
        'Realty agents': 'Administrative'
    })

    # Regroupement de ORGANIZATION_TYPE
    df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].apply(regrouper_organisation)

    return df

# Bureau

def nettoyer_colonnes_categorielles_bureau(df):
    """
    Nettoie et simplifie les colonnes catégorielles du fichier bureau.csv :
    - Supprime les lignes avec 'Bad debt' dans CREDIT_ACTIVE (trop rare)
    - Regroupe les valeurs rares ou équivalentes dans CREDIT_ACTIVE, CREDIT_CURRENCY et CREDIT_TYPE
    - Retourne le DataFrame nettoyé
    """

    df = df.copy()

    # Supprimer les lignes avec CREDIT_ACTIVE == 'Bad debt' (trop rare)
    df = df[df['CREDIT_ACTIVE'] != 'Bad debt']

    # Regrouper 'Sold' avec 'Closed'
    df['CREDIT_ACTIVE'] = df['CREDIT_ACTIVE'].replace({'Sold': 'Closed'})

    # CREDIT_CURRENCY : regrouper toutes les devises sauf 'currency 1' en 'Other'
    df['CREDIT_CURRENCY'] = df['CREDIT_CURRENCY'].apply(lambda x: x if x == 'currency 1' else 'Other')

    # CREDIT_TYPE : regrouper selon la logique discutée
    regroupement_credit_type = {
        'Credit card': 'Consumer',
        'Microloan': 'Consumer',
        'Cash loan (non-earmarked)': 'Consumer',
        'Car loan': 'Secured property',
        'Mortgage': 'Secured property',
        'Real estate loan': 'Secured property',
        'Loan for business development': 'Business',
        'Loan for working capital replenishment': 'Business',
        'Loan for the purchase of equipment': 'Business',
        'Loan for purchase of shares (margin lending)': 'Other',
        'Mobile operator loan': 'Other',
        'Interbank credit': 'Other',
        'Unknown type of loan': 'Other',
        'Another type of loan': 'Other'
    }

    df['CREDIT_TYPE'] = df['CREDIT_TYPE'].replace(regroupement_credit_type)

    return df

# Previous application 

def nettoyer_colonnes_categorielles_previous(df):
    """
    Nettoie et regroupe les colonnes catégorielles de previous_application
    pour réduire la cardinalité et supprimer les valeurs incohérentes
    sans supprimer de lignes ni introduire de NaN.
    """
    df = df.copy()

    # 🔁 Remplacer 'XNA' par 'Unknown' dans les colonnes suivantes
    colonnes_xna = [
        'NAME_CONTRACT_TYPE',
        'NAME_PAYMENT_TYPE',
        'NAME_CLIENT_TYPE',
        'NAME_PRODUCT_TYPE',
        'NAME_PORTFOLIO',
        'NAME_SELLER_INDUSTRY',
        'NAME_YIELD_GROUP'
    ]
    for col in colonnes_xna:
        if col in df.columns:
            df[col] = df[col].replace('XNA', 'Unknown')

    # 🔁 NAME_CASH_LOAN_PURPOSE : 'XNA' et 'XAP' → 'Unknown', rares → 'Other'
    df['NAME_CASH_LOAN_PURPOSE'] = df['NAME_CASH_LOAN_PURPOSE'].replace({'XNA': 'Unknown', 'XAP': 'Unknown'})
    rares = df['NAME_CASH_LOAN_PURPOSE'].value_counts()[df['NAME_CASH_LOAN_PURPOSE'].value_counts() < 1000].index
    df['NAME_CASH_LOAN_PURPOSE'] = df['NAME_CASH_LOAN_PURPOSE'].replace(rares, 'Other')

    # 🔁 NAME_GOODS_CATEGORY : 'XNA' → 'Unknown', rares → 'Other'
    df['NAME_GOODS_CATEGORY'] = df['NAME_GOODS_CATEGORY'].replace('XNA', 'Unknown')
    rares = df['NAME_GOODS_CATEGORY'].value_counts()[df['NAME_GOODS_CATEGORY'].value_counts() < 1000].index
    df['NAME_GOODS_CATEGORY'] = df['NAME_GOODS_CATEGORY'].replace(rares, 'Other')

    # 🔁 CHANNEL_TYPE : rares → 'Other'
    rares = df['CHANNEL_TYPE'].value_counts()[df['CHANNEL_TYPE'].value_counts() < 10000].index
    df['CHANNEL_TYPE'] = df['CHANNEL_TYPE'].replace(rares, 'Other')

    # 🔁 CODE_REJECT_REASON : regroupements logiques
    df['CODE_REJECT_REASON'] = df['CODE_REJECT_REASON'].replace({
        'XNA': 'Other',
        'XAP': 'Other',
        'HC': 'Client issue',
        'CLIENT': 'Client issue',
        'SCO': 'Scoring issue',
        'SCOFR': 'Scoring issue',
        'LIMIT': 'Credit limit',
        'VERIF': 'Technical',
        'SYSTEM': 'Technical'
    })

    # 🔁 WEEKDAY_APPR_PROCESS_START : regrouper les jours en semaine/week-end
    df['WEEKDAY_APPR_PROCESS_START'] = df['WEEKDAY_APPR_PROCESS_START'].replace({
        'SATURDAY': 'Weekend',
        'SUNDAY': 'Weekend',
        'MONDAY': 'Weekday',
        'TUESDAY': 'Weekday',
        'WEDNESDAY': 'Weekday',
        'THURSDAY': 'Weekday',
        'FRIDAY': 'Weekday'
    })

    # 🔁 NAME_CONTRACT_STATUS : pas de XNA, mais on peut regrouper "Unused offer" et "Canceled"
    df['NAME_CONTRACT_STATUS'] = df['NAME_CONTRACT_STATUS'].replace({
        'Unused offer': 'Canceled',
    })

    # 🔁 PRODUCT_COMBINATION : on peut simplifier les libellés en types généraux
    df['PRODUCT_COMBINATION'] = df['PRODUCT_COMBINATION'].replace({
        x: 'Cash' for x in df['PRODUCT_COMBINATION'].unique() if 'Cash' in str(x)
    })
    df['PRODUCT_COMBINATION'] = df['PRODUCT_COMBINATION'].replace({
        x: 'Card' for x in df['PRODUCT_COMBINATION'].unique() if 'Card' in str(x)
    })
    df['PRODUCT_COMBINATION'] = df['PRODUCT_COMBINATION'].replace({
        x: 'POS' for x in df['PRODUCT_COMBINATION'].unique() if 'POS' in str(x)
    })

    return df




