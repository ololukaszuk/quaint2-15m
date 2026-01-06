"""
Shared Constants
Configuration values used across services
"""

import os
from typing import List

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DB_HOST = os.getenv('DB_HOST', 'timescaledb')
DB_PORT = int(os.getenv('DB_PORT', '5432'))
DB_USER = os.getenv('DB_USER', 'mluser')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_NAME = os.getenv('DB_NAME', 'btc_ml')

# Database table names
TABLE_OHLCV_RAW = 'ohlcv_raw'
TABLE_OHLCV_15M = 'ohlcv_15m'
TABLE_TECHNICAL_INDICATORS = 'technical_indicators'
TABLE_ML_FEATURES = 'ml_features'
TABLE_PREDICTIONS = 'predictions'
TABLE_TRAINING_METADATA = 'training_metadata'
TABLE_MODEL_REGISTRY = 'model_registry'
TABLE_SYSTEM_HEALTH = 'system_health'

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model paths
MODEL_BASE_PATH = os.getenv('MODEL_PATH', '/app/models')
MODEL_ACTIVE_PATH = f"{MODEL_BASE_PATH}/active"
MODEL_ARCHIVE_PATH = f"{MODEL_BASE_PATH}/archive"

# Model filenames
LSTM_MODEL_FILENAME = 'lstm_model.pt'
XGB_MODEL_FILENAME = 'xgb_model.pkl'
NORMALIZER_FILENAME = 'normalizer.pkl'

# Model names
MODEL_A = 'ModelA'
MODEL_B = 'ModelB'
MODEL_C = 'Ensemble'

# LSTM configuration
LSTM_INPUT_SIZE = 1
LSTM_HIDDEN_SIZE = int(os.getenv('LSTM_HIDDEN_SIZE', '64'))
LSTM_NUM_LAYERS = int(os.getenv('LSTM_NUM_LAYERS', '2'))
LSTM_DROPOUT = float(os.getenv('LSTM_DROPOUT', '0.2'))
LSTM_LEARNING_RATE = float(os.getenv('LSTM_LEARNING_RATE', '0.001'))
LSTM_EPOCHS = int(os.getenv('LSTM_EPOCHS', '20'))
LSTM_BATCH_SIZE = int(os.getenv('LSTM_BATCH_SIZE', '64'))
LSTM_SEQUENCE_LENGTH = 60  # 60-minute lookback window

# XGBoost configuration
XGB_MAX_DEPTH = int(os.getenv('XGB_MAX_DEPTH', '6'))
XGB_N_ESTIMATORS = int(os.getenv('XGB_N_ESTIMATORS', '100'))
XGB_LEARNING_RATE = float(os.getenv('XGB_LEARNING_RATE', '0.1'))
XGB_MIN_CHILD_WEIGHT = 1
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_GAMMA = 0.1

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

# Technical indicators
INDICATOR_RSI_PERIOD = 14
INDICATOR_MACD_FAST = 12
INDICATOR_MACD_SLOW = 26
INDICATOR_MACD_SIGNAL = 9
INDICATOR_ATR_PERIOD = 14
INDICATOR_STOCH_K = 14
INDICATOR_STOCH_D = 3
INDICATOR_EMA_12 = 12
INDICATOR_EMA_26 = 26
INDICATOR_BOLLINGER_PERIOD = 20
INDICATOR_BOLLINGER_STD = 2.0
INDICATOR_VOLATILITY_WINDOW = 20

# Feature names (for ML input)
MOMENTUM_FEATURES: List[str] = [
    'rsi_14',
    'macd',
    'macd_hist',
    'stoch_k',
    'stoch_d'
]

VOLATILITY_FEATURES: List[str] = [
    'atr_14',
    'bb_width',
    'volatility_20'
]

TREND_FEATURES: List[str] = [
    'ema_12',
    'ema_26',
    'sma_20'
]

VOLUME_FEATURES: List[str] = [
    'obv',
    'volume_sma'
]

# All technical indicator features
TECHNICAL_FEATURES: List[str] = (
    MOMENTUM_FEATURES +
    VOLATILITY_FEATURES +
    TREND_FEATURES +
    VOLUME_FEATURES
)

# Total feature count (32 LSTM + len(TECHNICAL_FEATURES))
LSTM_LATENT_FEATURES = 32
TOTAL_FEATURE_COUNT = LSTM_LATENT_FEATURES + len(TECHNICAL_FEATURES)

# ============================================================================
# SERVICE CONFIGURATION
# ============================================================================

# Chainlink Poller
CHAINLINK_API_URL = os.getenv('CHAINLINK_API_URL', 'https://data.chain.link/streams/btc-usd')
CHAINLINK_FETCH_INTERVAL = int(os.getenv('CHAINLINK_FETCH_INTERVAL', '60'))  # seconds
CHAINLINK_MAX_RETRIES = int(os.getenv('CHAINLINK_MAX_RETRIES', '3'))
CHAINLINK_RETRY_DELAY = int(os.getenv('CHAINLINK_RETRY_DELAY', '5'))  # seconds

# Binance (OPTIONAL - for volume data only)
BINANCE_API_ENABLED = os.getenv('BINANCE_API_ENABLED', 'false').lower() == 'true'
BINANCE_API_BASE = os.getenv('BINANCE_API_BASE', 'https://api.binance.com/api/v3')
BINANCE_SYMBOL = os.getenv('BINANCE_SYMBOL', 'BTCUSDT')
BINANCE_INTERVAL = os.getenv('BINANCE_INTERVAL', '1m')

# Feature Engineer
FEATURE_CALCULATION_INTERVAL = int(os.getenv('FEATURE_CALCULATION_INTERVAL', '60'))  # seconds
FEATURE_LOOKBACK_MINUTES = int(os.getenv('FEATURE_LOOKBACK_MINUTES', '100'))

# Training Service
TRAINING_LOOKBACK_MONTHS = int(os.getenv('TRAINING_LOOKBACK_MONTHS', '12'))
TRAINING_SCHEDULE_CRON = os.getenv('TRAINING_SCHEDULE_CRON', '0 2 * * *')  # 02:00 UTC daily

# Inference Service
INFERENCE_INTERVAL = int(os.getenv('INFERENCE_INTERVAL', '60'))  # seconds
INFERENCE_LOOKBACK_MINUTES = int(os.getenv('INFERENCE_LOOKBACK_MINUTES', '60'))
INFERENCE_LATENCY_WARNING_THRESHOLD = float(os.getenv('INFERENCE_LATENCY_WARNING_THRESHOLD', '5.0'))  # seconds

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_DIR = os.getenv('LOG_DIR', '/app/logs')
LOG_MAX_SIZE_MB = int(os.getenv('LOG_MAX_SIZE_MB', '100'))
LOG_MAX_FILES = int(os.getenv('LOG_MAX_FILES', '10'))

# ============================================================================
# GPU/CUDA CONFIGURATION
# ============================================================================

CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', '0')

# ============================================================================
# PREDICTION CONFIGURATION
# ============================================================================

# Direction labels
DIRECTION_UP = 'UP'
DIRECTION_DOWN = 'DOWN'

# Confidence thresholds
MIN_CONFIDENCE = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.6
VERY_HIGH_CONFIDENCE_THRESHOLD = 0.7

# ============================================================================
# DATA QUALITY CONFIGURATION
# ============================================================================

# Missing data handling
MAX_MISSING_DATA_GAP_MINUTES = 10
MIN_CANDLES_FOR_INFERENCE = 30

# Outlier detection
OUTLIER_SIGMA_THRESHOLD = 3.0

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Data splits
TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.20
TEST_SPLIT = 0.10

# Early stopping
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001

# Walk-forward validation
WALK_FORWARD_FOLDS = 3

# ============================================================================
# MONITORING CONFIGURATION
# ============================================================================

ENABLE_METRICS = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
METRICS_PORT = int(os.getenv('METRICS_PORT', '9090'))

# Health check intervals
HEALTH_CHECK_INTERVAL_SECONDS = 60

# ============================================================================
# PATHS
# ============================================================================

# Base paths
BASE_PATH = '/app'
DATA_PATH = f"{BASE_PATH}/data"
LOGS_PATH = f"{BASE_PATH}/logs"
MODELS_PATH = f"{BASE_PATH}/models"

# ============================================================================
# API ENDPOINTS
# ============================================================================

# Binance (optional, for additional data)
BINANCE_API_BASE = 'https://api.binance.com/api/v3'
BINANCE_KLINES_ENDPOINT = f"{BINANCE_API_BASE}/klines"

# CoinGecko (optional, for additional metrics)
COINGECKO_API_BASE = 'https://api.coingecko.com/api/v3'

# ============================================================================
# TIMEOUTS
# ============================================================================

API_REQUEST_TIMEOUT = 10  # seconds
DATABASE_QUERY_TIMEOUT = 30  # seconds
MODEL_INFERENCE_TIMEOUT = 5  # seconds

# ============================================================================
# BATCH PROCESSING
# ============================================================================

BULK_INSERT_BATCH_SIZE = 1000
FEATURE_CALCULATION_BATCH_SIZE = 100

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_path(model_name: str, filename: str) -> str:
    """
    Get full path to model file.
    
    Args:
        model_name: Model identifier (ModelA, ModelB, etc.)
        filename: Model filename
        
    Returns:
        Full path to model file
    """
    return os.path.join(MODEL_ACTIVE_PATH, model_name, filename)


def get_archive_path(model_name: str, version: str) -> str:
    """
    Get path to archived model.
    
    Args:
        model_name: Model identifier
        version: Model version
        
    Returns:
        Full path to archive directory
    """
    return os.path.join(MODEL_ARCHIVE_PATH, f"{model_name}_{version}")


def validate_config() -> bool:
    """
    Validate that all required configuration values are set.
    
    Returns:
        True if configuration is valid
    
    Raises:
        ValueError: If required configuration is missing
    """
    required_vars = [
        ('DB_HOST', DB_HOST),
        ('DB_USER', DB_USER),
        ('DB_PASSWORD', DB_PASSWORD),
        ('DB_NAME', DB_NAME),
    ]
    
    missing = []
    for var_name, var_value in required_vars:
        if not var_value or var_value == '':
            missing.append(var_name)
    
    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")
    
    return True


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Validate configuration on import (optional, comment out if not desired)
# validate_config()
