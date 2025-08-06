import os
from celery import Celery
from celery.schedules import crontab

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

app = Celery('teera_reconciliation')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

# Celery Beat Schedule for periodic tasks
app.conf.beat_schedule = {
    'nightly-reconciliation-batch': {
        'task': 'reconciliation.tasks.nightly_reconciliation_batch',
        'schedule': crontab(hour=2, minute=0),  # Run at 2 AM daily
    },
    'retrain-ml-models': {
        'task': 'reconciliation.tasks.retrain_ml_model',
        'schedule': crontab(hour=3, minute=0, day_of_week=0),  # Run weekly on Sunday at 3 AM
    },
}

app.conf.timezone = 'UTC'

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
