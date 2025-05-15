# config.py
# Adjust paths/URLs here if needed

SQLITE_PATH        = "pipeline.db"
CELERY_BROKER_URL  = "redis://localhost:6379/0"
CELERY_RESULT_BACKEND = CELERY_BROKER_URL
DOMAIN_RULES_FILE  = "domain_rules.lp"
