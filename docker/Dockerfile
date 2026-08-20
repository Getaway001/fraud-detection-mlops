# Image unique réutilisée pour les trois étapes du pipeline
# (feature engineering, tuning, registry) : seule la commande lancée au
# runtime change (voir k8s/*.yaml et airflow/dags/), pas l'image elle-même.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Nécessaire pour que `python -m src.xxx.yyy` trouve le package src
# quel que soit le répertoire de travail effectif au runtime.
ENV PYTHONPATH=/app

# Commande par défaut (surchargée explicitement dans chaque Job/tâche Airflow).
CMD ["python", "-m", "src.training.train"]
