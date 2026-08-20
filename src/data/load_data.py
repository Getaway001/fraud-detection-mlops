"""
Chargement des données brutes et split train/test.

Le split est stratifié sur la cible (`Fraudulent`) pour garantir que la
proportion de fraudes (~4.9%) soit respectée dans les deux ensembles malgré
le déséquilibre de classes — sans stratification, un split aléatoire
pourrait produire un jeu de test avec très peu (voire aucun) exemple de
fraude, rendant l'évaluation peu fiable.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_raw_data(path: str = None) -> pd.DataFrame:
    """Charge le CSV brut depuis le chemin donné (ou config.RAW_DATA_PATH)."""
    path = path or config.RAW_DATA_PATH
    print(f"Chargement des données brutes depuis {path}")
    df = pd.read_csv(path)
    print(f"{len(df)} lignes chargées, {df.shape[1]} colonnes")
    return df


def split_data(df: pd.DataFrame):
    """Sépare features (X) et cible (y), puis split stratifié train/test."""
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    print(f"Train: {len(X_train)} lignes ({y_train.mean():.2%} de fraudes)")
    print(f"Test:  {len(X_test)} lignes ({y_test.mean():.2%} de fraudes)")

    return X_train, X_test, y_train, y_test
