import os
from celery import Celery
from .settings import CELERY_RESULT_BACKEND, BROKER_URL

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'best_solution.settings')

app = Celery('best_solution', backend=CELERY_RESULT_BACKEND, broker=BROKER_URL)
print(BROKER_URL)
app.config_from_object('django.conf:settings')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()
