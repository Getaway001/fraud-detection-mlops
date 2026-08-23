"""
Entraînement d'un modèle RandomForest sur les features déjà préparées.

Deux usages :
1. Exécuté directement (`python -m src.training.train`) : un seul run
   MLflow avec des hyperparamètres fixes (utile pour un test rapide, ou
   comme baseline avant de lancer une recherche Optuna).
2. Ses fonctions (`load_processed`, `train_and_log`, `compute_metrics`)
   sont réutilisées par `tune.py`, pour éviter de dupliquer la logique
   d'entraînement/évaluation entre le mode simple et le mode recherche.

Métriques choisies : le dataset est fortement déséquilibré (~4.9% de
fraudes), donc l'accuracy seule serait trompeuse (un modèle qui prédit
toujours "légitime" obtiendrait déjà ~95%). On logue aussi precision,
recall, F1 et l'AUC-PR (aire sous la courbe précision-rappel), plus adaptée
qu'un ROC-AUC classique sur des classes très déséquilibrées.
"""

import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

from src import config


def load_processed():
    """Charge les features déjà préparées par build_features.py."""
    train_df = pd.read_parquet(config.TRAIN_FEATURES_PATH)
    test_df = pd.read_parquet(config.TEST_FEATURES_PATH)

    X_train = train_df.drop(columns=[config.TARGET_COLUMN])
    y_train = train_df[config.TARGET_COLUMN]
    X_test = test_df.drop(columns=[config.TARGET_COLUMN])
    y_test = test_df[config.TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Calcule les métriques adaptées à une classification déséquilibrée."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_pr": average_precision_score(y_true, y_proba),
    }


def train_and_log(params: dict, X_train, X_test, y_train, y_test, log_model: bool = True):
    """
    Entraîne un RandomForest avec les hyperparamètres donnés, logue tout
    dans le run MLflow actif, et retourne le modèle + ses métriques.
    """
    mlflow.log_params(params)

    model = RandomForestClassifier(random_state=config.RANDOM_STATE, **params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    mlflow.log_metrics(metrics)

    if log_model:
        # input_example permet à MLflow d'inférer automatiquement une
        # signature (types/noms des colonnes attendues). Sans elle, KServe/
        # MLServer ne savent pas convertir une requête V2 en DataFrame
        # avant l'appel au modèle, et la prédiction échoue en production
        # (erreur observée : "float() argument ... not 'InferenceRequest'").
        mlflow.sklearn.log_model(model, "model", input_example=X_train.iloc[:5])

    return model, metrics


def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    # Hyperparamètres fixes pour ce mode simple, surchargeables via env vars.
    n_estimators = int(os.environ.get("N_ESTIMATORS", 200))
    max_depth_env = os.environ.get("MAX_DEPTH", "10")
    max_depth = None if max_depth_env.lower() == "none" else int(max_depth_env)
    class_weight = os.environ.get("CLASS_WEIGHT", "balanced")
    class_weight = None if class_weight.lower() == "none" else class_weight

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "class_weight": class_weight,
    }

    X_train, X_test, y_train, y_test = load_processed()

    with mlflow.start_run(run_name="single_run"):
        print(f"Run ID : {mlflow.active_run().info.run_id}")
        print(f"Hyperparamètres : {params}")

        _, metrics = train_and_log(params, X_train, X_test, y_train, y_test)

        for name, value in metrics.items():
            print(f"{name:>10} : {value:.4f}")

        print("Run terminé et loggé dans MLflow avec succès.")


if __name__ == "__main__":
    main()