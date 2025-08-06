
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ML_CONFIG = {

    'TRANSFORMER_MODEL': 'distilbert-base-uncased',
    'EMBEDDING_DIM': 256,
    'NUMERICAL_DIM': 5,
    'ATTENTION_HEADS': 8,
    'DROPOUT_RATE': 0.1,

    'BATCH_SIZE': 16,
    'LEARNING_RATE': 2e-5,
    'EPOCHS': 10,
    'WEIGHT_DECAY': 0.01,
    'GRAD_CLIP_NORM': 1.0,

    'MAX_TEXT_LENGTH': 128,
    'NEGATIVE_RATIO': 1.5,
    'VALIDATION_SPLIT': 0.2,
    'MIN_TRAINING_SAMPLES': 100,

    'DEFAULT_TOP_K': 5,
    'MIN_CONFIDENCE': 0.3,
    'HIGH_CONFIDENCE': 0.8,
    'EXACT_MATCH_THRESHOLD': 0.01,
    'CLOSE_MATCH_THRESHOLD': 0.05,

    'TEXT_FEATURES': {
        'REFERENCE_PATTERNS': [
            r'INV[-_]?\d+',
            r'REF[-_]?\d+', 
            r'ORDER[-_]?\d+',
            r'\b\d{6,}\b',
        ],
        'COMPANY_PATTERNS': [
            r'LTD\.?$',
            r'LLC\.?$',
            r'CORP\.?$',
            r'INC\.?$',
        ],
        'PAYMENT_KEYWORDS': [
            'PAYMENT', 'TRANSFER', 'DEPOSIT', 
            'WIRE', 'ACH', 'CHECK', 'CREDIT'
        ],
        'INVOICE_KEYWORDS': [
            'INVOICE', 'BILL', 'CHARGE', 
            'FEE', 'SERVICE', 'SUBSCRIPTION'
        ]
    },

    'MODEL_DIR': BASE_DIR / 'ml_models',
    'MODEL_FILENAME': 'reconciliation_model.pth',
    'BACKUP_MODELS': 3,

    'CACHE_PREDICTIONS': True,
    'CACHE_TIMEOUT': 1800,
    'CACHE_KEY_PREFIX': 'ml_reconciliation_',

    'ACCURACY_THRESHOLD': 0.85,
    'RETRAIN_THRESHOLD': 0.80,
    'MONITORING_INTERVAL': 86400,

    'USE_GPU': True,
    'NUM_WORKERS': 4,
    'PIN_MEMORY': True,

    'FEATURE_IMPORTANCE': True,
    'MODEL_INTERPRETABILITY': True,
    'UNCERTAINTY_ESTIMATION': False,
    'ONLINE_LEARNING': False,
}

if os.getenv('ML_ENV') == 'development':
    ML_CONFIG.update({
        'BATCH_SIZE': 8,
        'EPOCHS': 3,
        'MIN_TRAINING_SAMPLES': 50,
        'CACHE_TIMEOUT': 300,
    })

elif os.getenv('ML_ENV') == 'production':
    ML_CONFIG.update({
        'BATCH_SIZE': 32,
        'EPOCHS': 20,
        'MIN_TRAINING_SAMPLES': 500,
        'CACHE_TIMEOUT': 3600,
        'BACKUP_MODELS': 5,
    })

CUDA_CONFIG = {
    'DEVICE_ID': 0,
    'MEMORY_FRACTION': 0.8,
    'ALLOW_GROWTH': True,
}

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'ml_formatter': {
            'format': '[{asctime}] {levelname} {name}: {message}',
            'style': '{',
        },
    },
    'handlers': {
        'ml_file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': BASE_DIR / 'logs' / 'ml_engine.log',
            'maxBytes': 10485760,
            'backupCount': 5,
            'formatter': 'ml_formatter',
        },
        'ml_console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'ml_formatter',
        },
    },
    'loggers': {
        'ml_engine': {
            'handlers': ['ml_file', 'ml_console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

DATA_CONFIG = {
    'PREPROCESSING': {
        'TEXT_CLEANING': True,
        'NORMALIZE_AMOUNTS': True,
        'HANDLE_MISSING_DATES': True,
        'REMOVE_DUPLICATES': True,
    },

    'AUGMENTATION': {
        'ENABLED': True,
        'SYNONYM_REPLACEMENT': False,
        'RANDOM_DELETION': False,
        'NOISE_INJECTION': False,
    },

    'VALIDATION': {
        'CROSS_VALIDATION_FOLDS': 5,
        'STRATIFIED_SPLIT': True,
        'TEMPORAL_SPLIT': False,
    }
}

ENSEMBLE_CONFIG = {
    'ENABLED': False,
    'N_MODELS': 3,
    'VOTING_STRATEGY': 'soft',
    'DIVERSITY_WEIGHT': 0.1,
}

MONITORING_CONFIG = {
    'PERFORMANCE_TRACKING': True,
    'DRIFT_DETECTION': True,
    'ALERT_THRESHOLD': 0.05,
    'ALERT_EMAIL': None,
    'METRICS_STORAGE': 'database',
}

def get_ml_config():
    return ML_CONFIG

def get_model_path():
    model_dir = ML_CONFIG['MODEL_DIR']
    model_dir.mkdir(exist_ok=True)
    return model_dir / ML_CONFIG['MODEL_FILENAME']

def get_cache_key(prefix, *args):
    key_parts = [ML_CONFIG['CACHE_KEY_PREFIX'] + prefix] + [str(arg) for arg in args]
    return '_'.join(key_parts)

def get_model_version():
    import datetime
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def validate_config():
    errors = []

    if ML_CONFIG['BATCH_SIZE'] < 1:
        errors.append("BATCH_SIZE must be positive")

    if ML_CONFIG['LEARNING_RATE'] <= 0:
        errors.append("LEARNING_RATE must be positive")

    if ML_CONFIG['MIN_CONFIDENCE'] < 0 or ML_CONFIG['MIN_CONFIDENCE'] > 1:
        errors.append("MIN_CONFIDENCE must be between 0 and 1")

    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")

    return True

validate_config()
