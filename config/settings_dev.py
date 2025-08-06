# Development settings that override base settings
from .settings import *

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Debug settings
DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1']

# Disable some middleware for development
MIDDLEWARE = [m for m in MIDDLEWARE if 'corsheaders' not in m]

# Logging - less verbose for development
LOGGING['root']['level'] = 'DEBUG'
LOGGING['handlers']['console']['level'] = 'DEBUG'

# Celery - use synchronous execution for development
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True

# Email backend for development
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
