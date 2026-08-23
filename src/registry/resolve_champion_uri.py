"""
Résout l'URI S3 réelle du modèle actuellement enregistré sous l'alias
`@champion` dans le Model Registry MLflow.

Pourquoi c'est nécessaire : le Model Registry ne stocke pas le modèle
lui-même, seulement une référence vers son emplacement dans MLflow. Cette
référence utilise le schéma interne `mlflow-artifacts:/...` (le mode proxy
qu'on a activé sur le serveur MLflow), qui permet aux clients MLflow d'y
accéder via HTTP à travers le tracking server. Mais KServe, pour charger un
modèle directement depuis un stockage S3, a besoin d'une vraie URI
`s3://<bucket>/<chemin>` — pas de ce schéma interne.

Ce module convertit l'un vers l'autre : le "chemin dans le bucket" est
identique dans les deux cas (c'est littéralement la même donnée physique
dans MinIO), seul le préfixe change.

    mlflow-artifacts:/1/72348736.../artifacts/model
                     ↓ (même chemin, préfixe différent)
    s3://mlflow/1/72348736.../artifacts/model

Ce module expose une fonction pure (`mlflow_source_to_s3_uri`), testable
sans connexion réseau, et un point d'entrée CLI qui interroge le vrai
Registry pour afficher l'URI du champion actuel.
"""

import mlflow
from mlflow.tracking import MlflowClient

from src import config

MLFLOW_ARTIFACTS_SCHEME = "mlflow-artifacts:/"


def mlflow_source_to_s3_uri(source: str, bucket: str = None) -> str:
    """
    Convertit une URI source de ModelVersion MLflow en URI S3 exploitable
    directement par un outil externe (KServe, boto3, etc.).

    - Si `source` utilise déjà le schéma "s3://", elle est retournée telle
      quelle (cas où le stockage n'est pas proxifié).
    - Si `source` utilise le schéma interne "mlflow-artifacts:/", on la
      convertit vers "s3://<bucket>/<même chemin>".
    - Tout autre schéma lève une erreur explicite plutôt que de deviner.
    """
    bucket = bucket or config.MLFLOW_ARTIFACT_BUCKET

    if source.startswith("s3://"):
        return source

    if source.startswith(MLFLOW_ARTIFACTS_SCHEME):
        # "mlflow-artifacts:/1/abc.../artifacts/model" -> "1/abc.../artifacts/model"
        path_within_bucket = source[len(MLFLOW_ARTIFACTS_SCHEME):]
        return f"s3://{bucket}/{path_within_bucket}"

    raise ValueError(
        f"Schéma d'URI non géré : '{source}'. "
        "Attendu 's3://...' ou 'mlflow-artifacts:/...'. "
        "As-tu changé la configuration du stockage d'artefacts MLflow ?"
    )


def resolve_champion_s3_uri() -> str:
    """
    Interroge le Model Registry pour retrouver la version actuellement en
    `@champion`, et retourne son URI S3 réelle.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()

    model_version = client.get_model_version_by_alias(
        config.REGISTERED_MODEL_NAME, config.CHAMPION_ALIAS
    )

    print(f"Modèle : {config.REGISTERED_MODEL_NAME}")
    print(f"Version @{config.CHAMPION_ALIAS} : v{model_version.version}")
    print(f"Source MLflow : {model_version.source}")

    s3_uri = mlflow_source_to_s3_uri(model_version.source)
    print(f"URI S3 résolue : {s3_uri}")

    return s3_uri


def main():
    resolve_champion_s3_uri()


if __name__ == "__main__":
    main()