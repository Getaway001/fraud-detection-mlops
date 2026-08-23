"""
Envoie une requête de test au modèle déployé sur KServe (Open Inference
Protocol V2), en utilisant de vraies lignes du jeu de test préparé par le
feature engineering — pour valider que le service d'inférence fonctionne
de bout en bout (téléchargement du modèle, chargement par MLServer,
prédiction), et comparer les prédictions à la réalité.

Conçu pour tourner DANS le cluster (via un Job Kubernetes, voir
k8s/serving/test-inference-job.yaml), avec accès au PVC partagé où vivent
les features déjà transformées (data/processed/test.parquet) et au DNS
interne du service predictor.
"""

import os

import pandas as pd
import requests

from src import config

PREDICTOR_URL = os.environ.get(
    "PREDICTOR_URL",
    "http://fraud-detection-model-predictor.mlops.svc.cluster.local/v2/models/fraud-detection-model/infer",
)
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", 5))


def build_v2_payload(X: pd.DataFrame) -> dict:
    """
    Construit un payload conforme à l'Open Inference Protocol V2 : un input
    par colonne, nommé exactement comme la colonne du DataFrame, avec le
    type de données V2 correspondant au type pandas réel de la colonne.

    Fonctionne à condition que le modèle ait été loggé avec une signature
    MLflow (voir train.py, log_model(..., input_example=...)) — c'est elle
    qui permet à MLServer/MLflow de reconstruire correctement un DataFrame
    à partir de ces tenseurs nommés avant l'appel au modèle.

    Important : le type de chaque colonne doit correspondre exactement au
    schéma inféré par MLflow (ex: colonnes entières comme
    Previous_Fraudulent_Transactions doivent rester des entiers, pas être
    converties en float) — MLflow refuse une conversion float -> int jugée
    "non sûre" et lève une MlflowException si les types ne correspondent
    pas (voir historique de debug : "Can not safely convert float64 to
    int64").
    """
    inputs = []
    for column in X.columns:
        series = X[column]

        if pd.api.types.is_bool_dtype(series):
            datatype = "BOOL"
            values = series.astype(bool).tolist()
        elif pd.api.types.is_integer_dtype(series):
            datatype = "INT64"
            values = series.astype("int64").tolist()
        else:
            datatype = "FP64"
            values = series.astype("float64").tolist()

        inputs.append(
            {
                "name": column,
                "shape": [len(values)],
                "datatype": datatype,
                "data": values,
            }
        )
    return {"inputs": inputs}


def main():
    print(f"Chargement de {NUM_SAMPLES} échantillons depuis {config.TEST_FEATURES_PATH}")
    test_df = pd.read_parquet(config.TEST_FEATURES_PATH)

    y_true = test_df[config.TARGET_COLUMN]
    X = test_df.drop(columns=[config.TARGET_COLUMN])

    sample_X = X.iloc[:NUM_SAMPLES]
    sample_y = y_true.iloc[:NUM_SAMPLES].tolist()

    payload = build_v2_payload(sample_X)

    print(f"Envoi de la requête vers {PREDICTOR_URL}")
    response = requests.post(PREDICTOR_URL, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()

    predictions = result["outputs"][0]["data"]

    print("\nComparaison prédictions vs réalité :")
    for i, (pred, true) in enumerate(zip(predictions, sample_y)):
        match = "✓" if int(pred) == int(true) else "✗"
        print(f"  Échantillon {i} : prédit={pred} | réel={true}  {match}")

    print("\nService d'inférence fonctionnel de bout en bout.")


if __name__ == "__main__":
    main()