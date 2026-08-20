"""
Fonctions pures de feature engineering.

Chaque fonction prend un DataFrame en entrée et retourne un DataFrame (ou un
DataFrame + un objet d'état à réutiliser, ex: les médianes calculées sur le
train), sans dépendance à MLflow, Kubernetes, ni effet de bord caché.

C'est un choix délibéré : ces fonctions représentent la logique de
transformation "métier", indépendante de l'infrastructure. Le jour où un
vrai feature store (type Feast) sera mis en place, cette logique pourra
migrer telle quelle vers des "feature views", sans réécriture.
"""

import pandas as pd

from src import config


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les colonnes identifiants, non prédictives en l'état."""
    return df.drop(columns=[c for c in config.ID_COLUMNS if c in df.columns], errors="ignore")


def impute_numeric(df: pd.DataFrame, medians: pd.Series = None):
    """
    Impute les valeurs manquantes des colonnes numériques par la médiane.

    Si `medians` est fourni (calculé sur le train), on l'applique tel quel
    au lieu de recalculer — c'est essentiel pour ne jamais faire "fuiter"
    d'information du test vers le train (éviter le data leakage).
    """
    df = df.copy()
    if medians is None:
        medians = df[config.NUMERIC_COLUMNS].median()
    for col in config.NUMERIC_COLUMNS:
        df[col] = df[col].fillna(medians[col])
    return df, medians


def impute_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute les valeurs manquantes des colonnes catégorielles par la
    catégorie explicite "Unknown", plutôt que de les masquer — l'absence
    de valeur peut elle-même être une information utile au modèle.
    """
    df = df.copy()
    for col in config.CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("Unknown")
    return df


def encode_categorical(df: pd.DataFrame, reference_columns=None) -> pd.DataFrame:
    """
    One-hot encode les colonnes catégorielles.

    Si `reference_columns` est fourni (les colonnes obtenues sur le train),
    on aligne le résultat sur ces colonnes exactement — nécessaire pour que
    train et test aient toujours la même structure de colonnes, même si une
    catégorie rare n'apparaît que dans l'un des deux ensembles.
    """
    df = pd.get_dummies(df, columns=config.CATEGORICAL_COLUMNS, prefix=config.CATEGORICAL_COLUMNS)
    if reference_columns is not None:
        df = df.reindex(columns=reference_columns, fill_value=0)
    return df
