# celery_app.py
from celery import Celery
from config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

app = Celery(
    "nlp_asp_pipeline",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)
