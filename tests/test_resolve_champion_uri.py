"""
Tests unitaires de la conversion d'URI MLflow -> S3.

Ne nécessite aucune connexion réseau : on teste uniquement la logique de
transformation de chaîne (mlflow_source_to_s3_uri), pas l'appel réel au
Model Registry (resolve_champion_s3_uri, qui lui a besoin d'un serveur
MLflow actif).
"""

import pytest

from src.registry.resolve_champion_uri import mlflow_source_to_s3_uri


def test_converts_mlflow_artifacts_scheme_to_s3():
    source = "mlflow-artifacts:/1/72348736c97f4f648ce3519cf5778ab1/artifacts/model"
    result = mlflow_source_to_s3_uri(source, bucket="mlflow")
    assert result == "s3://mlflow/1/72348736c97f4f648ce3519cf5778ab1/artifacts/model"


def test_passes_through_existing_s3_uri_unchanged():
    source = "s3://mlflow/1/abc/artifacts/model"
    result = mlflow_source_to_s3_uri(source, bucket="mlflow")
    assert result == source


def test_uses_default_bucket_from_config_when_not_specified():
    source = "mlflow-artifacts:/2/xyz/artifacts/model"
    result = mlflow_source_to_s3_uri(source)  # pas de bucket explicite
    assert result.startswith("s3://")
    assert result.endswith("2/xyz/artifacts/model")


def test_raises_on_unknown_scheme():
    with pytest.raises(ValueError):
        mlflow_source_to_s3_uri("gs://some-bucket/path/model")