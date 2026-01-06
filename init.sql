-- BTC ML Prediction System - Database Schema
-- TimescaleDB (PostgreSQL 15+ with time-series extension)

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================
-- TABLE: ohlcv_raw
-- Purpose: Raw 1-minute OHLCV data from Chainlink (PRIMARY) and Binance (VOLUME)
-- CRITICAL: Chainlink is GROUND TRUTH for price (Polymarket settlement source)
--           Binance may be used for volume data only (features)
-- ============================================================================
CREATE TABLE IF NOT EXISTS ohlcv_raw (
    time TIMESTAMPTZ NOT NULL,
    open DECIMAL NOT NULL,
    high DECIMAL NOT NULL,
    low DECIMAL NOT NULL,
    close DECIMAL NOT NULL,
    volume DECIMAL NOT NULL DEFAULT 0,
    source VARCHAR(50) DEFAULT 'chainlink',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT check_valid_source CHECK (source IN ('chainlink', 'binance'))
);

-- Convert to hypertable (TimescaleDB feature)
SELECT create_hypertable('ohlcv_raw', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ohlcv_raw_time_desc ON ohlcv_raw (time DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_raw_source ON ohlcv_raw (source);

-- Unique constraint
CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_raw_time_unique ON ohlcv_raw (time);

-- ============================================================================
-- TABLE: ohlcv_15m
-- Purpose: Aggregated 15-minute candles (derived from ohlcv_raw)
-- ============================================================================
CREATE TABLE IF NOT EXISTS ohlcv_15m (
    time TIMESTAMPTZ NOT NULL,
    open DECIMAL NOT NULL,
    high DECIMAL NOT NULL,
    low DECIMAL NOT NULL,
    close DECIMAL NOT NULL,
    volume DECIMAL NOT NULL,
    vwap DECIMAL,
    num_trades INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('ohlcv_15m', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '7 days');

CREATE INDEX IF NOT EXISTS idx_ohlcv_15m_time_desc ON ohlcv_15m (time DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_15m_time_unique ON ohlcv_15m (time);

-- ============================================================================
-- TABLE: technical_indicators
-- Purpose: Calculated technical indicators for ML features
-- ============================================================================
CREATE TABLE IF NOT EXISTS technical_indicators (
    time TIMESTAMPTZ NOT NULL,
    
    -- Momentum Indicators
    rsi_14 DECIMAL,
    macd DECIMAL,
    macd_signal DECIMAL,
    macd_hist DECIMAL,
    stoch_k DECIMAL,
    stoch_d DECIMAL,
    
    -- Volatility Indicators
    atr_14 DECIMAL,
    bb_upper DECIMAL,
    bb_middle DECIMAL,
    bb_lower DECIMAL,
    bb_width DECIMAL,
    volatility_20 DECIMAL,
    
    -- Trend Indicators
    ema_12 DECIMAL,
    ema_26 DECIMAL,
    sma_20 DECIMAL,
    
    -- Volume Indicators
    obv DECIMAL,
    vpt DECIMAL,
    volume_sma DECIMAL,
    
    -- Additional
    poc_level DECIMAL,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('technical_indicators', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '7 days');

CREATE INDEX IF NOT EXISTS idx_tech_ind_time_desc ON technical_indicators (time DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tech_ind_time_unique ON technical_indicators (time);

-- ============================================================================
-- TABLE: ml_features
-- Purpose: Normalized ML-ready features (optional, for caching)
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_features (
    time TIMESTAMPTZ NOT NULL,
    feature_vector DOUBLE PRECISION[],
    feature_names TEXT[],
    normalization_params JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('ml_features', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '7 days');

CREATE INDEX IF NOT EXISTS idx_ml_features_time_desc ON ml_features (time DESC);

-- ============================================================================
-- TABLE: predictions
-- Purpose: Model inference outputs
-- ============================================================================
CREATE TABLE IF NOT EXISTS predictions (
    time TIMESTAMPTZ NOT NULL,
    direction VARCHAR(10) NOT NULL CHECK (direction IN ('UP', 'DOWN')),
    confidence DECIMAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    model_name VARCHAR(50) NOT NULL,
    lstm_latent DOUBLE PRECISION[],
    xgb_probabilities JSONB,
    latency_ms INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('predictions', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');

CREATE INDEX IF NOT EXISTS idx_predictions_time_desc ON predictions (time DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions (model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_direction ON predictions (direction);

-- ============================================================================
-- TABLE: training_metadata
-- Purpose: Track model training runs and metrics
-- ============================================================================
CREATE TABLE IF NOT EXISTS training_metadata (
    trained_at TIMESTAMPTZ NOT NULL PRIMARY KEY,
    model_a_accuracy DECIMAL,
    model_b_accuracy DECIMAL,
    ensemble_accuracy DECIMAL,
    validation_auc_roc DECIMAL,
    training_duration_seconds INT,
    n_samples_trained INT,
    model_a_path VARCHAR(255),
    model_b_path VARCHAR(255),
    lstm_params JSONB,
    xgb_params JSONB,
    feature_importance JSONB,
    validation_metrics JSONB,
    test_metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_metadata_trained_at_desc ON training_metadata (trained_at DESC);

-- ============================================================================
-- TABLE: model_registry
-- Purpose: Track active and archived models
-- ============================================================================
CREATE TABLE IF NOT EXISTS model_registry (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('active', 'archived', 'testing')),
    lstm_path VARCHAR(255),
    xgb_path VARCHAR(255),
    accuracy DECIMAL,
    precision_score DECIMAL,
    recall_score DECIMAL,
    f1_score DECIMAL,
    auc_roc DECIMAL,
    trained_at TIMESTAMPTZ NOT NULL,
    deployed_at TIMESTAMPTZ,
    archived_at TIMESTAMPTZ,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_registry_status ON model_registry (status);
CREATE INDEX IF NOT EXISTS idx_model_registry_name ON model_registry (model_name);
CREATE INDEX IF NOT EXISTS idx_model_registry_trained_at ON model_registry (trained_at DESC);

-- ============================================================================
-- TABLE: system_health
-- Purpose: Track service health and monitoring metrics
-- ============================================================================
CREATE TABLE IF NOT EXISTS system_health (
    check_time TIMESTAMPTZ NOT NULL,
    service_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) CHECK (status IN ('healthy', 'degraded', 'down')),
    latency_ms INT,
    error_message TEXT,
    additional_info JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('system_health', 'check_time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');

CREATE INDEX IF NOT EXISTS idx_system_health_time_desc ON system_health (check_time DESC);
CREATE INDEX IF NOT EXISTS idx_system_health_service ON system_health (service_name);

-- ============================================================================
-- CONTINUOUS AGGREGATES (TimescaleDB feature)
-- Purpose: Pre-computed aggregations for faster queries
-- ============================================================================

-- Hourly aggregates from 1-minute data
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    FIRST(open, time) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, time) AS close,
    SUM(volume) AS volume
FROM ohlcv_raw
GROUP BY bucket;

-- Daily prediction summary
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_prediction_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS bucket,
    model_name,
    COUNT(*) AS total_predictions,
    SUM(CASE WHEN direction = 'UP' THEN 1 ELSE 0 END) AS up_count,
    SUM(CASE WHEN direction = 'DOWN' THEN 1 ELSE 0 END) AS down_count,
    AVG(confidence) AS avg_confidence,
    MIN(confidence) AS min_confidence,
    MAX(confidence) AS max_confidence
FROM predictions
GROUP BY bucket, model_name;

-- ============================================================================
-- RETENTION POLICIES
-- Purpose: Automatic data cleanup to manage storage
-- ============================================================================

-- Keep raw 1-min data for 90 days
SELECT add_retention_policy('ohlcv_raw', INTERVAL '90 days', if_not_exists => TRUE);

-- Keep predictions for 180 days
SELECT add_retention_policy('predictions', INTERVAL '180 days', if_not_exists => TRUE);

-- Keep system health for 30 days
SELECT add_retention_policy('system_health', INTERVAL '30 days', if_not_exists => TRUE);

-- Keep indicators for 180 days
SELECT add_retention_policy('technical_indicators', INTERVAL '180 days', if_not_exists => TRUE);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to get latest price
CREATE OR REPLACE FUNCTION get_latest_price()
RETURNS TABLE (
    time TIMESTAMPTZ,
    price DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT ohlcv_raw.time, ohlcv_raw.close
    FROM ohlcv_raw
    ORDER BY ohlcv_raw.time DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate accuracy over time window
CREATE OR REPLACE FUNCTION calculate_prediction_accuracy(
    model_name_param VARCHAR,
    hours_back INT DEFAULT 24
)
RETURNS TABLE (
    total_predictions BIGINT,
    avg_confidence DECIMAL,
    up_predictions BIGINT,
    down_predictions BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*) AS total_predictions,
        AVG(confidence)::DECIMAL AS avg_confidence,
        SUM(CASE WHEN direction = 'UP' THEN 1 ELSE 0 END) AS up_predictions,
        SUM(CASE WHEN direction = 'DOWN' THEN 1 ELSE 0 END) AS down_predictions
    FROM predictions
    WHERE 
        predictions.model_name = model_name_param
        AND predictions.time > NOW() - (hours_back || ' hours')::INTERVAL;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- GRANTS (adjust user as needed)
-- ============================================================================

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mluser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mluser;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO mluser;

-- ============================================================================
-- INITIAL DATA / SEED (Optional)
-- ============================================================================

-- Insert initial model registry entry (placeholder)
INSERT INTO model_registry (
    model_name, 
    version, 
    status, 
    lstm_path, 
    xgb_path, 
    trained_at
) VALUES (
    'ModelA',
    'v1.0.0',
    'testing',
    '/app/models/active/lstm_model.pt',
    '/app/models/active/xgb_model.pkl',
    NOW()
) ON CONFLICT DO NOTHING;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Uncomment to test schema creation
-- SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
-- SELECT * FROM timescaledb_information.hypertables;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
