"""
DAG orchestrant le pipeline complet de détection de fraude :

    feature_engineering  ->  hyperparameter_tuning  ->  register_best_model
        ->  deploy_champion_model

Chaque étape est un pod Kubernetes éphémère (KubernetesPodOperator), créé à
la demande par le KubernetesExecutor. Les deux premières étapes partagent
un PersistentVolumeClaim (`fraud-data-pvc`) pour faire transiter les
données : `feature_engineering` y écrit les features, `hyperparameter_tuning`
les relit. `register_best_model` et `deploy_champion_model` ne parlent
qu'aux API MLflow/Kubernetes, pas besoin du volume.

`deploy_champion_model` ferme la boucle MLOps : si `register_best_model` a
promu un nouveau `@champion`, cette tâche met automatiquement à jour
l'InferenceService KServe pour qu'il serve ce nouveau modèle (rolling
update automatique) — plus besoin d'intervention manuelle entre une
promotion de modèle et son déploiement effectif.

Déclenchement manuel pour l'instant (schedule=None), le temps de valider le
pipeline de bout en bout. Voir le README pour la logique détaillée de
chaque étape.
"""

from datetime import datetime

from airflow.sdk import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

NAMESPACE = "mlops"
IMAGE = "ghcr.io/getaway001/fraud-detection-mlops:latest"

MLFLOW_TRACKING_URI = "http://mlflow.mlops.svc.cluster.local"

# Volume partagé entre feature_engineering et hyperparameter_tuning.
DATA_VOLUME = k8s.V1Volume(
    name="data",
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
        claim_name="fraud-data-pvc"
    ),
)
DATA_VOLUME_MOUNT = k8s.V1VolumeMount(name="data", mount_path="/app/data")


with DAG(
    dag_id="fraud_detection_pipeline",
    description="Feature engineering -> Optuna tuning -> Model Registry -> Deploy",
    schedule=None,  # déclenchement manuel, le temps de valider le pipeline
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "mlflow", "fraud-detection"],
) as dag:

    feature_engineering = KubernetesPodOperator(
        task_id="feature_engineering",
        name="fraud-feature-engineering",
        namespace=NAMESPACE,
        image=IMAGE,
        image_pull_policy="Always",
        cmds=["python", "-m", "src.features.build_features"],
        env_vars={
            "RAW_DATA_PATH": "/app/data/raw/Fraud_Detection_Dataset.csv",
            "PROCESSED_DIR": "/app/data/processed",
        },
        volumes=[DATA_VOLUME],
        volume_mounts=[DATA_VOLUME_MOUNT],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    hyperparameter_tuning = KubernetesPodOperator(
        task_id="hyperparameter_tuning",
        name="fraud-hyperparameter-tuning",
        namespace=NAMESPACE,
        image=IMAGE,
        image_pull_policy="Always",
        cmds=["python", "-m", "src.training.tune"],
        env_vars={
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "PROCESSED_DIR": "/app/data/processed",
            "OPTUNA_N_TRIALS": "30",
            "OPTUNA_METRIC": "f1",
        },
        volumes=[DATA_VOLUME],
        volume_mounts=[DATA_VOLUME_MOUNT],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    register_best_model = KubernetesPodOperator(
        task_id="register_best_model",
        name="fraud-register-best-model",
        namespace=NAMESPACE,
        image=IMAGE,
        image_pull_policy="Always",
        cmds=["python", "-m", "src.registry.register_model"],
        env_vars={
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "OPTUNA_METRIC": "f1",
        },
        get_logs=True,
        is_delete_operator_pod=True,
    )

    deploy_champion_model = KubernetesPodOperator(
        task_id="deploy_champion_model",
        name="fraud-deploy-champion",
        namespace=NAMESPACE,
        image=IMAGE,
        image_pull_policy="Always",
        # ServiceAccount dédié, avec les permissions RBAC pour lire/modifier
        # l'InferenceService KServe (voir k8s/serving/deploy-rbac.yaml).
        service_account_name="fraud-deploy-sa",
        cmds=["python", "-m", "src.registry.deploy_champion"],
        env_vars={
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
        },
        get_logs=True,
        is_delete_operator_pod=True,
    )

    feature_engineering >> hyperparameter_tuning >> register_best_model >> deploy_champion_model