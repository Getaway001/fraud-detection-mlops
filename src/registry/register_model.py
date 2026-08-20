"""
Sélection du meilleur run et enregistrement dans le Model Registry MLflow.

Étape volontairement séparée du tuning (tune.py) : elle permet d'inspecter
les résultats d'une étude avant de décider de "promouvoir" un modèle,
plutôt que de l'enregistrer automatiquement dès la fin de l'entraînement.

Comportement : à chaque exécution, le run parent d'étude Optuna le plus
récent est enregistré comme une NOUVELLE VERSION du modèle dans le
Registry (l'historique de tous les candidats est conservé). L'alias
`@champion`, lui, n'est déplacé vers cette nouvelle version QUE si elle
est réellement meilleure que le champion actuel sur la métrique cible.
S'il n'y a pas encore de champion (premier enregistrement), la nouvelle
version est promue automatiquement.

Ça évite qu'une étude moins bien réglée (moins d'essais, mauvaise plage
d'hyperparamètres...) n'écrase silencieusement un meilleur modèle
précédent.
"""

import os

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src import config


def get_best_tuning_run(client: MlflowClient, experiment_id: str, metric_name: str):
    """
    Retrouve le run parent d'étude Optuna le plus récent dans l'expérience,
    identifié par le tag `run_type=optuna_tuning_parent` posé par tune.py.

    Trié par la métrique `best_<metric_name>` loguée sur ce run parent (le
    meilleur score obtenu par l'étude), le run le plus performant en tête.
    """
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.run_type = 'optuna_tuning_parent'",
        order_by=[f"metrics.best_{metric_name} DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(
            "Aucun run d'étude Optuna trouvé dans l'expérience "
            f"'{config.EXPERIMENT_NAME}' (tag run_type='optuna_tuning_parent'). "
            "As-tu bien lancé src.training.tune avant cette étape ?"
        )
    return runs[0]


def get_current_champion_score(client: MlflowClient, metric_name: str):
    """
    Retourne (version, score) du modèle actuellement en `@champion`, ou
    (None, None) si aucun champion n'existe encore (premier enregistrement).
    """
    try:
        champion_version = client.get_model_version_by_alias(
            config.REGISTERED_MODEL_NAME, config.CHAMPION_ALIAS
        )
    except MlflowException:
        # L'alias n'existe pas encore — cas normal au tout premier run.
        return None, None

    champion_run = client.get_run(champion_version.run_id)
    score = champion_run.data.metrics.get(f"best_{metric_name}")
    return champion_version.version, score


def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(config.EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Expérience '{config.EXPERIMENT_NAME}' introuvable.")

    metric_name = os.environ.get("OPTUNA_METRIC", config.OPTUNA_METRIC)

    # --- 1. Identifier le meilleur run de la dernière étude ---
    best_run = get_best_tuning_run(client, experiment.experiment_id, metric_name)
    run_id = best_run.info.run_id
    candidate_score = best_run.data.metrics.get(f"best_{metric_name}")

    print(f"Meilleur run d'étude trouvé : {run_id}")
    print(f"{metric_name} (candidat) = {candidate_score:.4f}")

    # --- 2. Enregistrer ce modèle comme nouvelle version (toujours) ---
    # On garde une trace de tous les candidats dans l'historique du Registry,
    # même ceux qui ne deviendront pas champion — utile pour comparer plus
    # tard ou revenir en arrière manuellement si besoin.
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(
        model_uri=model_uri, name=config.REGISTERED_MODEL_NAME
    )
    print(
        f"Modèle enregistré : '{config.REGISTERED_MODEL_NAME}' "
        f"version {registered_model.version} (candidat)"
    )
    client.set_model_version_tag(
        name=config.REGISTERED_MODEL_NAME,
        version=registered_model.version,
        key=metric_name,
        value=str(candidate_score),
    )

    # --- 3. Comparer au champion actuel avant de déplacer l'alias ---
    current_version, current_score = get_current_champion_score(client, metric_name)

    if current_score is None:
        promote = True
        reason = "aucun champion existant pour l'instant"
    elif candidate_score > current_score:
        promote = True
        reason = (
            f"meilleur que le champion actuel "
            f"(v{current_version}, {metric_name}={current_score:.4f})"
        )
    else:
        promote = False
        reason = (
            f"pas meilleur que le champion actuel "
            f"(v{current_version}, {metric_name}={current_score:.4f})"
        )

    if promote:
        client.set_registered_model_alias(
            name=config.REGISTERED_MODEL_NAME,
            alias=config.CHAMPION_ALIAS,
            version=registered_model.version,
        )
        print(
            f"Alias '@{config.CHAMPION_ALIAS}' déplacé vers la version "
            f"{registered_model.version} — {reason}"
        )
        print(
            "Chargement possible via : "
            f"models:/{config.REGISTERED_MODEL_NAME}@{config.CHAMPION_ALIAS}"
        )
    else:
        print(
            f"Alias '@{config.CHAMPION_ALIAS}' INCHANGÉ (reste sur la version "
            f"{current_version}) — {reason}"
        )
        print(
            f"La version {registered_model.version} reste enregistrée dans le "
            "Registry, mais n'est pas promue champion."
        )


if __name__ == "__main__":
    main()
