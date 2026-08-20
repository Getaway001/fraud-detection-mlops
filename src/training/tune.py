"""
Recherche d'hyperparamètres avec Optuna pour le modèle de détection de
fraude.

Un run MLflow parent (`fraud_detection_tuning`) représente l'étude Optuna
complète. Chaque essai (trial) est un run enfant imbriqué, loguant ses
hyperparamètres et métriques. Optuna oriente ses essais suivants en
fonction des résultats précédents (recherche bayésienne, sampler TPE) —
plus efficace qu'une grille exhaustive pour explorer un espace
d'hyperparamètres à 5 dimensions.

La métrique optimisée (config.OPTUNA_METRIC, F1 par défaut) est choisie
pour rester pertinente malgré le déséquilibre de classes du dataset — voir
train.py pour le détail du raisonnement.

Le run parent enregistre aussi le meilleur modèle trouvé, en artefact
(`mlflow.sklearn.log_model`). L'enregistrement dans le Model Registry
lui-même est fait dans une étape séparée (voir registry/register_model.py)
— une séparation volontaire pour permettre une validation entre
l'expérimentation et la promotion en "champion".
"""

import mlflow
import mlflow.sklearn
import optuna

from src import config
from src.training.train import load_processed, train_and_log


def objective(trial, X_train, X_test, y_train, y_test):
    """Un essai : Optuna propose des hyperparamètres, on entraîne et évalue."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        _, metrics = train_and_log(
            params, X_train, X_test, y_train, y_test, log_model=False
        )
        mlflow.set_tag("optuna_trial_number", trial.number)

    print(
        f"Trial {trial.number:>3} | {params} "
        f"| {config.OPTUNA_METRIC}={metrics[config.OPTUNA_METRIC]:.4f}"
    )

    return metrics[config.OPTUNA_METRIC]


def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    print(f"Connexion à MLflow sur {config.MLFLOW_TRACKING_URI}")
    print(f"Expérience : {config.EXPERIMENT_NAME}")
    print(f"Nombre d'essais Optuna : {config.OPTUNA_N_TRIALS}")
    print(f"Métrique optimisée : {config.OPTUNA_METRIC}\n")

    X_train, X_test, y_train, y_test = load_processed()

    with mlflow.start_run(run_name="fraud_detection_tuning") as parent_run:
        # Tag utilisé par register_model.py pour retrouver ce run parmi
        # tous les runs de l'expérience.
        mlflow.set_tag("run_type", "optuna_tuning_parent")
        mlflow.log_param("n_trials", config.OPTUNA_N_TRIALS)
        mlflow.log_param("optimize_metric", config.OPTUNA_METRIC)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
        )
        study.optimize(
            lambda trial: objective(trial, X_train, X_test, y_train, y_test),
            n_trials=config.OPTUNA_N_TRIALS,
        )

        best_params = study.best_params
        best_value = study.best_value

        print(
            f"\nMeilleurs hyperparamètres : {best_params} "
            f"({config.OPTUNA_METRIC}={best_value:.4f})"
        )
        print(f"Trouvés au trial n°{study.best_trial.number}")

        # Préfixe "best_" pour ne pas entrer en conflit avec les noms de
        # paramètres du run parent lui-même (n_trials, optimize_metric).
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric(f"best_{config.OPTUNA_METRIC}", best_value)
        mlflow.log_metric("best_trial_number", study.best_trial.number)

        # Ré-entraînement du meilleur modèle, loggé en artefact sur le run
        # parent — c'est ce modèle que register_model.py ira chercher.
        _, final_metrics = train_and_log(
            best_params, X_train, X_test, y_train, y_test, log_model=True
        )
        for name, value in final_metrics.items():
            mlflow.log_metric(f"final_{name}", value)

        print(f"\nRun parent MLflow : {parent_run.info.run_id}")
        print("Étude Optuna terminée et loguée dans MLflow avec succès.")


if __name__ == "__main__":
    main()
