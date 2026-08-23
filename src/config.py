"""
Configuration centralisée du projet.

Toutes les valeurs modifiables (chemins, noms MLflow, colonnes du dataset,
hyperparamètres par défaut) sont regroupées ici plutôt que dispersées dans
chaque script — un seul endroit à modifier si le dataset ou l'infrastructure
évolue.

Chaque constante peut être surchargée via une variable d'environnement, ce
qui permet de garder le même code entre l'exécution locale et les Jobs
Kubernetes (où les valeurs sont injectées via `env:` dans les manifests).
"""

import os

# --- MLflow ---
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "http://mlflow.mlops.svc.cluster.local"
)
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "fraud-detection")
REGISTERED_MODEL_NAME = os.environ.get("REGISTERED_MODEL_NAME", "fraud-detection-model")
CHAMPION_ALIAS = "champion"

# Bucket MinIO utilisé par MLflow pour stocker les artefacts (modèles, etc.),
# configuré dans chart-upgrades/values-artifacts.yaml (artifactRoot.s3.bucket).
# Nécessaire pour convertir un chemin "mlflow-artifacts:/..." (schéma proxy
# interne à MLflow) en une vraie URI S3 ("s3://...") utilisable directement
# par des outils externes comme KServe.
MLFLOW_ARTIFACT_BUCKET = os.environ.get("MLFLOW_ARTIFACT_BUCKET", "mlflow")

# --- Données ---
# En local (développement), ces chemins pointent vers le dossier du projet.
# Sur Kubernetes, ils pointent vers le PersistentVolumeClaim partagé monté
# dans chaque Job (voir k8s/*.yaml), ce qui permet à la tâche de feature
# engineering d'écrire des données que la tâche de tuning peut relire.
RAW_DATA_PATH = os.environ.get(
    "RAW_DATA_PATH", "data/raw/Fraud_Detection_Dataset.csv"
)
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "data/processed")
TRAIN_FEATURES_PATH = os.path.join(PROCESSED_DIR, "train.parquet")
TEST_FEATURES_PATH = os.path.join(PROCESSED_DIR, "test.parquet")

# --- Colonnes du dataset ---
TARGET_COLUMN = "Fraudulent"

# Identifiants : non prédictifs en l'état, exclus des features. Deviendront
# pertinents plus tard pour du feature engineering agrégé par utilisateur
# une fois un feature store en place.
ID_COLUMNS = ["Transaction_ID", "User_ID"]

NUMERIC_COLUMNS = [
    "Transaction_Amount",
    "Time_of_Transaction",
    "Previous_Fraudulent_Transactions",
    "Account_Age",
    "Number_of_Transactions_Last_24H",
]

CATEGORICAL_COLUMNS = [
    "Transaction_Type",
    "Device_Used",
    "Location",
    "Payment_Method",
]

# --- Split train/test ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Optuna ---
OPTUNA_N_TRIALS = int(os.environ.get("OPTUNA_N_TRIALS", 30))
# Métrique optimisée par Optuna. F1 plutôt qu'accuracy : le dataset est très
# déséquilibré (~4.9% de fraudes), un modèle qui prédit toujours "légitime"
# obtiendrait déjà ~95% d'accuracy sans détecter aucune fraude.
OPTUNA_METRIC = os.environ.get("OPTUNA_METRIC", "f1")