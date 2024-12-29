from __future__ import absolute_import, unicode_literals
# This will make sure the app is always imported when
# Django starts so Celery can discover tasks in this app.
from . import tasks