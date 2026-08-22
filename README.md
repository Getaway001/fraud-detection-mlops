# Fraud Detection MLOps

**Auteur :** Ibrahima DIOP

Détection de transactions frauduleuses à partir d'un dataset Kaggle, avec un
pipeline MLOps complet : ingestion des données, feature engineering,
entraînement avec recherche d'hyperparamètres (Optuna), sélection et
enregistrement du meilleur modèle dans le Model Registry MLflow — le tout
orchestré par Airflow sur Kubernetes (minikube).

## Contexte du dataset

- **Source** : Kaggle, `Fraud_Detection_Dataset.csv`
- **Volume** : 51 000 transactions, 12 colonnes
- **Cible** : `Fraudulent` (0 = légitime, 1 = frauduleuse) — **fortement
  déséquilibrée** (~4.9% de fraudes), ce qui impacte le choix des métriques
  (privilégier précision/rappel/F1/AUC-PR plutôt que l'accuracy seule) et la
  stratégie d'entraînement (`class_weight`, stratification du split).
- **Valeurs manquantes** sur plusieurs colonnes (~5% chacune) :
  `Transaction_Amount`, `Time_of_Transaction`, `Device_Used`, `Location`,
  `Payment_Method`.
- **Variables catégorielles** à encoder : `Transaction_Type`, `Device_Used`,
  `Location`, `Payment_Method`.

## Architecture générale

```
data/raw (CSV Kaggle)
        │
        ▼
 [1] Feature Engineering  ──────────────► data/processed (features prêtes)
        │
        ▼
 [2] Hyperparameter Tuning (Optuna)  ───► MLflow (runs imbriqués : 1 run
        │                                 parent "étude" + N runs enfants
        │                                 "essais", chacun avec ses params
        │                                 et métriques)
        ▼
 [3] Sélection + Model Registry  ───────► MLflow Model Registry
                                           (nouvelle version enregistrée,
                                            alias @champion mis à jour)
```

Chaque étape tourne comme un **Job Kubernetes** indépendant dans le
namespace `mlops`, orchestré par un **DAG Airflow** qui enchaîne les trois
étapes dans l'ordre, avec les dépendances explicites entre elles.

---

## Étape 1 — Upload / ingestion des données

**Objectif** : rendre le dataset brut disponible pour les étapes suivantes,
de façon reproductible (pas de dépendance à un fichier local sur une seule
machine), et faire transiter les données entre les Jobs Kubernetes
successifs (chaque Job est un pod éphémère, sans stockage partagé par
défaut).

**Approche actuelle : PersistentVolumeClaim partagé**

Pour rester simple pendant l'apprentissage, les trois premières étapes
(`feature_engineering`, `hyperparameter_tuning`) partagent un même
`PersistentVolumeClaim` (`fraud-data-pvc`, voir `k8s/data-pvc.yaml`) monté
sur `/app/data` dans chaque pod :
- `feature_engineering` y écrit `data/processed/{train,test}.parquet`
- `hyperparameter_tuning` les relit depuis ce même volume

**Ce qui se passe concrètement :**
1. Créer le PVC (une seule fois) :
   ```powershell
   kubectl apply -f k8s/data-pvc.yaml
   ```
2. Copier le CSV brut dedans, via un pod utilitaire temporaire :
   ```powershell
   kubectl apply -f k8s/data-loader-pod.yaml
   kubectl cp data/raw/Fraud_Detection_Dataset.csv `
     mlops/fraud-data-loader:/app/data/raw/Fraud_Detection_Dataset.csv
   kubectl delete pod fraud-data-loader -n mlops
   ```
3. Les Jobs suivants montent ce même PVC et lisent/écrivent directement
   dessus — aucune autre manipulation nécessaire ensuite.

**Limite connue** : `ReadWriteOnce` ne permet qu'à un seul nœud d'écrire à
la fois — suffisant sur minikube (un seul nœud), mais à revoir pour un
cluster multi-nœuds en production.

**Évolution prévue** : cette approche par PVC est une étape intermédiaire.
Le stockage MinIO déjà en place pour les artefacts MLflow (bucket
`mlflow`) pourra héberger aussi les données brutes/transformées (bucket
dédié `fraud-data`), ce qui lèvera la limite `ReadWriteOnce` et préparera
la migration vers un vrai offline feature store.

---

## Étape 2 — Feature Engineering

**Objectif** : transformer les données brutes en un jeu de features propre,
prêt pour l'entraînement.

**Ce qui se passe (`src/features/`) :**
1. **Chargement** des données brutes depuis MinIO (ou `data/raw/` en local
   pour du développement rapide).
2. **Imputation des valeurs manquantes** :
   - Colonnes numériques (`Transaction_Amount`, `Time_of_Transaction`) :
     imputation par médiane (robuste aux valeurs extrêmes, fréquentes dans
     des montants de transaction).
   - Colonnes catégorielles (`Device_Used`, `Location`, `Payment_Method`) :
     imputation par une catégorie explicite `"Unknown"` — préserve
     l'information "donnée manquante" plutôt que de l'masquer.
3. **Encodage des variables catégorielles** (`Transaction_Type`,
   `Device_Used`, `Location`, `Payment_Method`) — one-hot encoding, avec
   gestion des catégories jamais vues à l'inférence.
4. **Exclusion des identifiants** (`Transaction_ID`, `User_ID`) des features
   d'entraînement — non prédictifs en l'état (deviendront pertinents plus
   tard pour des features agrégées par utilisateur, une fois le feature
   store en place).
5. **Split train/test stratifié** sur la cible `Fraudulent`, pour garantir
   que la proportion de fraudes soit respectée dans les deux ensembles
   malgré le déséquilibre de classes.
6. **Sauvegarde** du résultat dans `data/processed/` (et/ou MinIO) — ce
   dossier joue le rôle d'un offline feature store rudimentaire en
   attendant l'intégration d'un vrai outil (Feast ou équivalent).

**Sortie** : deux jeux de données (train/test), prêts à être consommés tels
quels par n'importe quel run d'entraînement, sans retraitement.

---

## Étape 3 — Training & recherche des meilleurs hyperparamètres (Optuna)

**Objectif** : trouver la meilleure configuration de modèle pour ce
problème de classification déséquilibrée, en explorant intelligemment
l'espace des hyperparamètres plutôt qu'à l'aveugle.

**Ce qui se passe (`src/training/tune.py`) :**
1. Un **run MLflow parent** (`fraud_detection_tuning`) représente l'étude
   Optuna complète.
2. Optuna lance N essais (`trials`), chacun testant une combinaison
   d'hyperparamètres (ex : profondeur d'arbre, nombre d'estimateurs, taux
   d'apprentissage selon le modèle choisi) choisie de façon bayésienne en
   fonction des essais précédents.
3. **Chaque essai est un run MLflow enfant** (imbriqué sous le parent),
   loguant :
   - Les hyperparamètres testés
   - Les métriques adaptées au déséquilibre de classes : précision,
     rappel, F1-score, AUC-PR (plus informatif que l'accuracy sur une
     cible à 4.9% de positifs)
4. La métrique d'optimisation (celle qu'Optuna cherche à maximiser) est
   définie explicitement — typiquement le F1-score ou l'AUC-PR plutôt que
   l'accuracy, pour éviter qu'Optuna ne converge vers un modèle qui
   prédit "pas de fraude" en permanence (ce qui donnerait déjà 95%
   d'accuracy sans détecter aucune fraude).
5. À la fin de l'étude, le run parent logue les meilleurs hyperparamètres
   trouvés, ré-entraîne un modèle avec cette configuration, et logue ce
   modèle comme artefact — **sans encore l'enregistrer dans le Registry**
   (étape suivante, volontairement séparée).

**Pourquoi séparer tuning et registry** : ça permet d'inspecter/valider les
résultats d'une étude avant de décider de "promouvoir" un modèle — une
étape de contrôle qualité entre l'expérimentation et la mise en production,
que ce soit automatique ou manuel selon le contexte.

---

## Étape 4 — Sélection du meilleur modèle & Model Registry

**Objectif** : identifier objectivement le meilleur run parmi une étude
terminée, et le rendre accessible sous un nom stable pour la suite
(serving, déploiement).

**Ce qui se passe (`src/registry/register_model.py`) :**
1. Interroge MLflow pour retrouver, au sein de l'expérience concernée, le
   run parent d'étude Optuna le plus récent (tag `run_type=optuna_tuning_parent`)
   et sa meilleure valeur de métrique (`best_<metric>`, ex : F1-score).
2. **Enregistre systématiquement** ce modèle comme une nouvelle version dans
   le **Model Registry** (ex : `fraud-detection-model`) — l'historique de
   tous les candidats est conservé, même ceux qui ne deviennent pas
   champion, ce qui permet de comparer ou revenir en arrière manuellement.
3. **Compare** le score de ce candidat à celui du modèle actuellement en
   alias `@champion` :
   - Si **aucun champion n'existe encore** (premier enregistrement), le
     candidat est promu automatiquement.
   - Si le candidat est **meilleur** que le champion actuel sur la
     métrique cible, l'alias `@champion` est déplacé vers cette nouvelle
     version.
   - Sinon, l'alias **reste inchangé** — la nouvelle version est
     enregistrée dans le Registry mais n'est pas promue, ce qui évite
     qu'une étude moins bien réglée n'écrase silencieusement un meilleur
     modèle précédent.
4. Le code de serving peut toujours charger
   `models:/fraud-detection-model@champion` sans jamais avoir à changer,
   quelle que soit la version qui se trouve effectivement derrière cet
   alias à un instant donné.

**Sortie** : une version numérotée et nommée du modèle, prête à être
chargée depuis n'importe quel environnement ayant accès à MLflow, sans
connaître de Run ID technique.

---

## Orchestration : le DAG Airflow

Le DAG `fraud_detection_pipeline` enchaîne les trois étapes ci-dessus comme
des tâches Kubernetes indépendantes (`KubernetesPodOperator`), avec des
dépendances explicites entre elles.

```
   ┌───────────────────────┐
   │  feature_engineering  │   Transforme data/raw → data/processed
   └───────────┬───────────┘
               │
               ▼
   ┌───────────────────────┐
   │   hyperparameter_      │   Étude Optuna, N runs MLflow imbriqués
   │        tuning          │
   └───────────┬───────────┘
               │
               ▼
   ┌───────────────────────┐
   │   register_best_       │   Sélection + enregistrement dans le
   │        model            │   Model Registry, mise à jour @champion
   └───────────────────────┘
```

**Détail des tâches :**

| Tâche | Opérateur | Rôle | Dépend de |
|---|---|---|---|
| `feature_engineering` | `KubernetesPodOperator` | Lit `data/raw`, produit `data/processed` | — |
| `hyperparameter_tuning` | `KubernetesPodOperator` | Étude Optuna, logging MLflow | `feature_engineering` |
| `register_best_model` | `KubernetesPodOperator` | Sélectionne et enregistre le meilleur modèle | `hyperparameter_tuning` |

**Planification** : déclenchement manuel dans un premier temps
(`schedule=None`), le temps de valider le pipeline de bout en bout.
Passage à un planning récurrent (ex : hebdomadaire, pour ré-entraîner sur
de nouvelles données) envisageable une fois validé.

**Visibilité dans l'UI Airflow** : chaque tâche apparaît comme un nœud
distinct dans le graphe du DAG, avec son statut (succès/échec) et ses
logs consultables directement, sans repasser par `kubectl logs`.

---

## Développement local (sans Kubernetes)

Pour itérer rapidement sur le code sans passer par des pods à chaque fois :

```bash
pip install -r requirements.txt

# Feature engineering (lit data/raw/, écrit data/processed/)
python -m src.features.build_features

# Entraînement simple (un seul run)
python -m src.training.train

# Recherche Optuna (nécessite MLflow accessible, ex: via port-forward)
MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.training.tune

# Tests unitaires du feature engineering
pytest tests/
```

## Prérequis

- minikube démarré avec le driver `docker`, ressources suffisantes
  (≥ 8 Go RAM recommandé vu MLflow + Postgres + MinIO + Airflow)
- MLflow déployé dans le namespace `mlops` (Postgres + MinIO configurés)
- Airflow déployé dans le namespace `mlops` (`KubernetesExecutor`)
- Docker installé en local pour builder l'image du projet

## Démarrage rapide

Depuis la mise en place du CI/CD (GitHub Actions), l'image Docker est
automatiquement buildée et publiée sur GitHub Container Registry
(`ghcr.io/getaway001/fraud-detection-mlops:latest`) à chaque push sur
`main` — plus besoin de builder l'image manuellement en local ni de la
charger dans minikube.

```powershell
# 0. Créer le PVC partagé et y charger le dataset brut (une seule fois)
kubectl apply -f k8s/data-pvc.yaml
kubectl apply -f k8s/data-loader-pod.yaml
kubectl cp data/raw/Fraud_Detection_Dataset.csv `
  mlops/fraud-data-loader:/app/data/raw/Fraud_Detection_Dataset.csv
kubectl delete pod fraud-data-loader -n mlops

# 1. Test manuel étape par étape (optionnel, avant de passer par Airflow)
#    Les Jobs utilisent directement l'image publiée sur GHCR.
kubectl apply -f k8s/feature-engineering-job.yaml
kubectl logs -n mlops -l step=feature-engineering -f

kubectl apply -f k8s/tuning-job.yaml
kubectl logs -n mlops -l step=tuning -f

kubectl apply -f k8s/register-model-job.yaml
kubectl logs -n mlops -l step=register-model -f

# 2. Déclenchement depuis l'UI Airflow (http://localhost:8080)
#    Le DAG est synchronisé automatiquement depuis Git (gitSync), plus
#    besoin de kubectl cp pour le déployer.
#    Déclenchement en ligne de commande :
kubectl exec -it <pod-scheduler> -n mlops -c scheduler -- `
  airflow dags trigger fraud_detection_pipeline
```

**Note sur `imagePullPolicy: Always`** : chaque exécution retélécharge la
dernière image publiée par le CI/CD — pratique en apprentissage pour
toujours avoir le code à jour, mais à reconsidérer en production (préférer
un tag versionné explicite plutôt que `:latest`, pour la reproductibilité).

## Structure du projet

```
fraud-detection-mlops/
├── data/
│   ├── raw/                 # dataset brut (non versionné)
│   └── processed/           # features transformées
├── src/
│   ├── config.py
│   ├── data/                # chargement + split
│   ├── features/            # feature engineering (transformations pures)
│   ├── training/            # entraînement simple + tuning Optuna
│   └── registry/            # sélection + enregistrement Model Registry
├── docker/
│   └── Dockerfile
├── k8s/                     # manifests des Jobs (pour tests manuels hors Airflow)
├── airflow/dags/
│   └── fraud_detection_pipeline_dag.py
├── chart-upgrades/           # values-*.yaml MLflow / Airflow
├── notebooks/                # exploration ponctuelle (EDA)
├── tests/
├── requirements.txt
└── README.md
```

## Prochaines étapes envisagées

- Intégration d'un vrai feature store (offline + online), ex : Feast
- Comparaison automatique avant remplacement de l'alias `@champion`
- Déploiement du modèle `@champion` comme service d'inférence
- Planification récurrente du DAG (ré-entraînement périodique)
- Détection de dérive des données (data drift) déclenchant un
  ré-entraînement automatique
