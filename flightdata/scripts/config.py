# config.py - Centralized configuration for Flight Price Prediction Project

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data files
RAW_DATA_FILE = "../data/Clean_Dataset.csv"
PROCESSED_DATA_FILE = "flight_data_processed.csv"
TRAIN_TEST_FILES = {
    'X_train': 'X_train.csv',
    'X_test': 'X_test.csv', 
    'y_train': 'y_train.csv',
    'y_test': 'y_test.csv'
}

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
PCA_VARIANCE_THRESHOLD = 0.95
ANOMALY_CONTAMINATION = 0.05

# Model hyperparameters (optimized for faster training)
HYPERPARAMETERS = {
    'xgboost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1], 
        'max_depth': [3, 6],
        'subsample': [0.8, 1.0]
    },
    'lightgbm': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 6]
    },
    'catboost': {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.1],
        'depth': [3, 6]
    },
    'decision_tree': {
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
}

# Anomaly detection settings
ANOMALY_MODELS = {
    'isolation_forest': {'contamination': 0.05, 'random_state': RANDOM_STATE},
    'one_class_svm': {'nu': 0.05},
    'dbscan': {'eps': 0.5, 'min_samples': 10}
}

# API settings
API_HOST = '0.0.0.0'
API_PORT = 5000
API_DEBUG = True

# Feature engineering settings
CATEGORICAL_FEATURES = [
    'airline', 'source_city', 'departure_time', 
    'arrival_time', 'destination_city', 'class'
]

NUMERICAL_FEATURES = ['duration', 'days_left', 'price']

STOPS_MAPPING = {'zero': 0, 'one': 1, 'two_or_more': 2}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': str(LOGS_DIR / 'flight_prediction.log'),
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}
