"""
Deep Learning Configuration for Bank Reconciliation
===================================================

This file contains all configuration settings for the ML engine.
Modify these settings to tune model performance and behavior.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Model Configuration
ML_CONFIG = {
    # Model Architecture
    'TRANSFORMER_MODEL': 'distilbert-base-uncased',  # Lightweight BERT model
    'EMBEDDING_DIM': 256,                            # Text embedding dimension
    'NUMERICAL_DIM': 5,                             # Number of numerical features
    'ATTENTION_HEADS': 8,                           # Multi-head attention heads
    'DROPOUT_RATE': 0.1,                           # Dropout for regularization
    
    # Training Parameters
    'BATCH_SIZE': 16,                               # Training batch size
    'LEARNING_RATE': 2e-5,                         # Adam learning rate
    'EPOCHS': 10,                                  # Default training epochs
    'WEIGHT_DECAY': 0.01,                         # L2 regularization
    'GRAD_CLIP_NORM': 1.0,                        # Gradient clipping
    
    # Data Configuration
    'MAX_TEXT_LENGTH': 128,                        # Maximum tokenization length
    'NEGATIVE_RATIO': 1.5,                        # Negative to positive sample ratio
    'VALIDATION_SPLIT': 0.2,                      # Validation data percentage
    'MIN_TRAINING_SAMPLES': 100,                  # Minimum samples for training
    
    # Prediction Parameters
    'DEFAULT_TOP_K': 5,                           # Default number of suggestions
    'MIN_CONFIDENCE': 0.3,                       # Minimum confidence threshold
    'HIGH_CONFIDENCE': 0.8,                      # High confidence threshold
    'EXACT_MATCH_THRESHOLD': 0.01,               # Amount difference for exact match
    'CLOSE_MATCH_THRESHOLD': 0.05,               # Percentage for close match
    
    # Feature Engineering
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
    
    # Model Storage
    'MODEL_DIR': BASE_DIR / 'ml_models',
    'MODEL_FILENAME': 'reconciliation_model.pth',
    'BACKUP_MODELS': 3,                           # Number of model backups to keep
    
    # Caching
    'CACHE_PREDICTIONS': True,                    # Cache ML predictions
    'CACHE_TIMEOUT': 1800,                       # Cache timeout in seconds (30 min)
    'CACHE_KEY_PREFIX': 'ml_reconciliation_',     # Cache key prefix
    
    # Performance Monitoring
    'ACCURACY_THRESHOLD': 0.85,                  # Minimum acceptable accuracy
    'RETRAIN_THRESHOLD': 0.80,                   # Retrain if accuracy drops below
    'MONITORING_INTERVAL': 86400,                # Performance check interval (24h)
    
    # Hardware Settings
    'USE_GPU': True,                             # Use GPU if available
    'NUM_WORKERS': 4,                            # DataLoader worker processes
    'PIN_MEMORY': True,                          # Pin memory for faster GPU transfer
    
    # Advanced Features
    'FEATURE_IMPORTANCE': True,                   # Track feature importance
    'MODEL_INTERPRETABILITY': True,              # Enable model explanations
    'UNCERTAINTY_ESTIMATION': False,             # Bayesian uncertainty (experimental)
    'ONLINE_LEARNING': False,                    # Incremental learning (experimental)
}

# Environment-specific overrides
if os.getenv('ML_ENV') == 'development':
    ML_CONFIG.update({
        'BATCH_SIZE': 8,
        'EPOCHS': 3,
        'MIN_TRAINING_SAMPLES': 50,
        'CACHE_TIMEOUT': 300,  # 5 minutes in development
    })

elif os.getenv('ML_ENV') == 'production':
    ML_CONFIG.update({
        'BATCH_SIZE': 32,
        'EPOCHS': 20,
        'MIN_TRAINING_SAMPLES': 500,
        'CACHE_TIMEOUT': 3600,  # 1 hour in production
        'BACKUP_MODELS': 5,
    })

# GPU Configuration
CUDA_CONFIG = {
    'DEVICE_ID': 0,                              # Default GPU device
    'MEMORY_FRACTION': 0.8,                     # Fraction of GPU memory to use
    'ALLOW_GROWTH': True,                       # Allow dynamic memory allocation
}

# Logging Configuration
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
            'maxBytes': 10485760,  # 10MB
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

# Data Pipeline Configuration
DATA_CONFIG = {
    'PREPROCESSING': {
        'TEXT_CLEANING': True,                   # Clean text before processing
        'NORMALIZE_AMOUNTS': True,               # Normalize amount ranges
        'HANDLE_MISSING_DATES': True,           # Fill missing dates
        'REMOVE_DUPLICATES': True,              # Remove duplicate records
    },
    
    'AUGMENTATION': {
        'ENABLED': True,                        # Enable data augmentation
        'SYNONYM_REPLACEMENT': False,           # Replace words with synonyms
        'RANDOM_DELETION': False,               # Randomly delete words
        'NOISE_INJECTION': False,               # Add noise to numerical features
    },
    
    'VALIDATION': {
        'CROSS_VALIDATION_FOLDS': 5,           # K-fold cross validation
        'STRATIFIED_SPLIT': True,               # Maintain class balance
        'TEMPORAL_SPLIT': False,                # Split by time for time series
    }
}

# Model Ensemble Configuration (Advanced)
ENSEMBLE_CONFIG = {
    'ENABLED': False,                           # Enable model ensemble
    'N_MODELS': 3,                             # Number of models in ensemble
    'VOTING_STRATEGY': 'soft',                 # 'hard' or 'soft' voting
    'DIVERSITY_WEIGHT': 0.1,                   # Weight for model diversity
}

# Monitoring and Alerts
MONITORING_CONFIG = {
    'PERFORMANCE_TRACKING': True,               # Track model performance
    'DRIFT_DETECTION': True,                   # Detect data/concept drift
    'ALERT_THRESHOLD': 0.05,                   # Performance drop alert threshold
    'ALERT_EMAIL': None,                       # Email for alerts (set in production)
    'METRICS_STORAGE': 'database',             # 'database' or 'file'
}

# Export configuration
def get_ml_config():
    """Get the complete ML configuration."""
    return ML_CONFIG

def get_model_path():
    """Get the full path to the model file."""
    model_dir = ML_CONFIG['MODEL_DIR']
    model_dir.mkdir(exist_ok=True)
    return model_dir / ML_CONFIG['MODEL_FILENAME']

def get_cache_key(prefix, *args):
    """Generate cache key for ML operations."""
    key_parts = [ML_CONFIG['CACHE_KEY_PREFIX'] + prefix] + [str(arg) for arg in args]
    return '_'.join(key_parts)

# Model versioning
def get_model_version():
    """Get current model version."""
    import datetime
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# Configuration validation
def validate_config():
    """Validate ML configuration settings."""
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

# Initialize configuration on import
validate_config()
