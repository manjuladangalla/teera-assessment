
from .settings import *

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1']

MIDDLEWARE = [m for m in MIDDLEWARE if 'corsheaders' not in m]

LOGGING['root']['level'] = 'DEBUG'
LOGGING['handlers']['console']['level'] = 'DEBUG'

CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True

EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
