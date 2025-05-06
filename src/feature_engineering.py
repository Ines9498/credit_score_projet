import pandas as pd
import numpy as np


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def feature_engineering_bureau(bureau_df):
    bureau_df, bureau_cat = one_hot_encoder(bureau_df)

    num_agg = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'CNT_CREDIT_PROLONG': ['sum']
    }
    cat_agg = {cat: ['mean'] for cat in bureau_cat}

    bureau_agg = bureau_df.groupby('SK_ID_CURR').agg({**num_agg, **cat_agg})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + '_' + e[1].upper() for e in bureau_agg.columns.tolist()])
    return bureau_agg.reset_index()


def feature_engineering_previous(previous_df):
    previous_df, prev_cat = one_hot_encoder(previous_df)

    # Vérifier l'existence des colonnes avant agrégation
    num_agg_cols = [
        'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT',
        'AMT_GOODS_PRICE', 'APP_CREDIT_PERC', 'HOUR_APPR_PROCESS_START',
        'DAYS_DECISION', 'CNT_PAYMENT'
    ]
    num_agg = {col: ['min', 'max', 'mean'] for col in num_agg_cols if col in previous_df.columns}

    # Ajouter les colonnes spécifiques si elles existent
    if 'AMT_DOWN_PAYMENT' in previous_df.columns:
        num_agg['AMT_DOWN_PAYMENT'] = ['min', 'max', 'mean']
    if 'RATE_DOWN_PAYMENT' in previous_df.columns:
        num_agg['RATE_DOWN_PAYMENT'] = ['min', 'max', 'mean']

    if 'AMT_APPLICATION' in previous_df.columns and 'AMT_CREDIT' in previous_df.columns:
        previous_df['APP_CREDIT_PERC'] = previous_df['AMT_APPLICATION'] / previous_df['AMT_CREDIT']

    cat_agg = {cat: ['mean'] for cat in prev_cat}

    previous_agg = previous_df.groupby('SK_ID_CURR').agg({**num_agg, **cat_agg})
    previous_agg.columns = pd.Index(['PREV_' + e[0] + '_' + e[1].upper() for e in previous_agg.columns.tolist()])
    return previous_agg.reset_index()


def fusionner_et_agreger_donnees(application_df, bureau_df, previous_df):
    # Feature engineering
    bureau_agg = feature_engineering_bureau(bureau_df)
    previous_agg = feature_engineering_previous(previous_df)

    # Fusion
    application_df = application_df.merge(bureau_agg, how='left', on='SK_ID_CURR')
    application_df = application_df.merge(previous_agg, how='left', on='SK_ID_CURR')

    return application_df
