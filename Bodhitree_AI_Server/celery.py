# from __future__ import absolute_import, unicode_literals
# import os
# from celery import Celery
# from django.conf import settings
# from celery.schedules import crontab  # Import crontab for scheduling tasks

# # Set the default Django settings module for the 'celery' program.
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Bodhitree_AI_Server.settings')

# app = Celery('Bodhitree_AI_Server')

# # Load task modules from all registered Django app configs.
# app.config_from_object('django.conf:settings', namespace='CELERY')
# # app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)

# app.autodiscover_tasks()

# # Define Celery beat schedule
# app.conf.beat_schedule = {
#     'retarining_process': {
#         'task': 'Datacollector.tasks.handle_retraining_process',  # Use the correct path to your task
#         # 'schedule': crontab(hour=0, minute=0),  # Executes daily at midnight
#         'schedule': crontab(minute='*/2'),
#     },
# }

# @app.task(bind=True)
# def debug_task(self):
#     print(f'Request: {self.request!r}')


from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
from celery.schedules import crontab

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Bodhitree_AI_Server.settings')

app = Celery('Bodhitree_AI_Server')

# Load task modules from all registered Django app configs.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Remove the settings import and use app.autodiscover_tasks() without arguments
app.autodiscover_tasks()

# Define Celery beat schedule
app.conf.beat_schedule = {
    'retarining_process': {
        'task': 'Datacollector.tasks.handle_retraining_process',
        'schedule': crontab(minute='*/20'),
    },
}

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')