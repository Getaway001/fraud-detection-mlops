"""
Orchestration du feature engineering.

Enchaîne : chargement des données brutes -> split train/test -> imputation
-> encodage -> sauvegarde des features prêtes à l'entraînement.

Les statistiques nécessaires à la transformation (médianes, colonnes
encodées) sont calculées uniquement sur le train, puis réappliquées telles
quelles au test — c'est la règle à respecter pour éviter le data leakage
(le test ne doit jamais influencer les statistiques utilisées pour le
transformer).

Point de stockage : pour l'instant, les features sont écrites sur le
PersistentVolumeClaim partagé entre les Jobs Kubernetes (voir k8s/*.yaml).
Ce dossier `data/processed/` joue le rôle d'un offline feature store
rudimentaire, en attendant l'intégration d'un vrai outil (Feast ou
équivalent) — l'interface (lire des features déjà calculées) restera la
même quand cette migration aura lieu.
"""

import os

from src import config
from src.data.load_data import load_raw_data, split_data
from src.features.transformations import (
    drop_id_columns,
    impute_numeric,
    impute_categorical,
    encode_categorical,
)


def build_features():
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # --- Transformations appliquées au train (statistiques calculées ici) ---
    X_train = drop_id_columns(X_train)
    X_train, medians = impute_numeric(X_train)
    X_train = impute_categorical(X_train)
    X_train_encoded = encode_categorical(X_train)

    # --- Mêmes transformations appliquées au test, avec les stats du train ---
    X_test = drop_id_columns(X_test)
    X_test, _ = impute_numeric(X_test, medians=medians)
    X_test = impute_categorical(X_test)
    X_test_encoded = encode_categorical(X_test, reference_columns=X_train_encoded.columns)

    # --- Réassemblage features + cible, sauvegarde ---
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    train_out = X_train_encoded.copy()
    train_out[config.TARGET_COLUMN] = y_train.values

    test_out = X_test_encoded.copy()
    test_out[config.TARGET_COLUMN] = y_test.values

    train_out.to_parquet(config.TRAIN_FEATURES_PATH, index=False)
    test_out.to_parquet(config.TEST_FEATURES_PATH, index=False)

    print(f"Features train : {train_out.shape} -> {config.TRAIN_FEATURES_PATH}")
    print(f"Features test  : {test_out.shape} -> {config.TEST_FEATURES_PATH}")
    print("Feature engineering terminé.")


if __name__ == "__main__":
    build_features()
