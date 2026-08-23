"""
Synchronise automatiquement l'InferenceService KServe avec le modèle
actuellement en @champion dans le Model Registry MLflow.

Avant ce script, faire pointer KServe vers un nouveau @champion nécessitait
trois étapes manuelles : résoudre l'URI S3 (resolve_champion_uri.py),
éditer le manifeste YAML, puis `kubectl apply`. Ce script automatise
l'ensemble, pour qu'un déploiement suive chaque promotion de modèle sans
intervention humaine.

Idempotent : si l'InferenceService pointe déjà vers l'URI du champion
actuel, aucune modification n'est faite (évite un rolling update inutile
à chaque exécution du pipeline si le champion n'a pas changé).

Nécessite un ServiceAccount avec les permissions RBAC pour lire/modifier
les InferenceService du namespace (voir k8s/serving/deploy-rbac.yaml).
"""

import os

from kubernetes import client, config as k8s_config

from src.registry.resolve_champion_uri import resolve_champion_s3_uri

NAMESPACE = os.environ.get("KSERVE_NAMESPACE", "mlops")
INFERENCE_SERVICE_NAME = os.environ.get("INFERENCE_SERVICE_NAME", "fraud-detection-model")

# Coordonnées du CRD KServe InferenceService (groupe/version/pluriel),
# nécessaires pour l'API Kubernetes générique des ressources personnalisées.
GROUP = "serving.kserve.io"
API_VERSION = "v1beta1"
PLURAL = "inferenceservices"


def get_current_storage_uri(api: client.CustomObjectsApi) -> str:
    """Lit le storageUri actuellement configuré sur l'InferenceService."""
    isvc = api.get_namespaced_custom_object(
        GROUP, API_VERSION, NAMESPACE, PLURAL, INFERENCE_SERVICE_NAME
    )
    return isvc["spec"]["predictor"]["model"]["storageUri"]


def patch_storage_uri(api: client.CustomObjectsApi, new_uri: str) -> None:
    """Met à jour le storageUri — déclenche un rolling update côté KServe."""
    patch_body = {"spec": {"predictor": {"model": {"storageUri": new_uri}}}}
    api.patch_namespaced_custom_object(
        GROUP, API_VERSION, NAMESPACE, PLURAL, INFERENCE_SERVICE_NAME, patch_body
    )


def main():
    print("Résolution de l'URI S3 du modèle @champion actuel...")
    champion_uri = resolve_champion_s3_uri()

    # Charge la configuration Kubernetes depuis l'intérieur du pod (via le
    # ServiceAccount monté automatiquement) — pas besoin de kubeconfig.
    k8s_config.load_incluster_config()
    api = client.CustomObjectsApi()

    current_uri = get_current_storage_uri(api)
    print(f"\nstorageUri actuel de l'InferenceService : {current_uri}")
    print(f"storageUri du champion actuel            : {champion_uri}")

    if current_uri == champion_uri:
        print("\nLe service pointe déjà vers le modèle @champion actuel. Rien à faire.")
        return

    print("\nNouveau champion détecté — mise à jour de l'InferenceService...")
    patch_storage_uri(api, champion_uri)
    print(
        "InferenceService mis à jour avec succès. KServe va recréer le pod "
        "predictor pour charger le nouveau modèle (rolling update)."
    )


if __name__ == "__main__":
    main()