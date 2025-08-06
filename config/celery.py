import os
from celery import Celery
from celery.schedules import crontab

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

app = Celery('teera_reconciliation')

app.config_from_object('django.conf:settings', namespace='CELERY')

app.autodiscover_tasks()

app.conf.beat_schedule = {
    'nightly-reconciliation-batch': {
        'task': 'reconciliation.tasks.nightly_reconciliation_batch',
        'schedule': crontab(hour=2, minute=0),
    },
    'retrain-ml-models': {
        'task': 'reconciliation.tasks.retrain_ml_model',
        'schedule': crontab(hour=3, minute=0, day_of_week=0),
    },
}

app.conf.timezone = 'UTC'

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
