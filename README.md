# BTC ML Prediction System

Production-grade Bitcoin price direction prediction system using hybrid LSTM + XGBoost architecture.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MICROSERVICES ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────┘

TimescaleDB (PostgreSQL + Time-Series)
    ↓
Chainlink Poller → Feature Engineer → Training Service → Inference Service
    ↓                    ↓                  ↓                  ↓
  (Raw OHLCV)      (Indicators)        (Models)         (Predictions)
```

## Tech Stack

- **Python**: 3.11
- **ML Framework**: PyTorch 2.0+ (LSTM), XGBoost 2.0+ (Gradient Boosting)
- **Database**: TimescaleDB (PostgreSQL 15+ with time-series extension)
- **Orchestration**: Docker Compose
- **GPU**: CUDA 12.8 (RTX 5090 compatible)

## Data Sources

### ⚠️ CRITICAL: Why Chainlink is Primary

This system is designed for **Polymarket predictions**, which uses **Chainlink BTC/USD** as the official settlement source. Therefore:

**PRICE (Ground Truth)**: Chainlink BTC/USD Data Streams
- **Used for**: All predictions, labels, inference targets
- **Why**: Polymarket uses Chainlink to determine market settlement
- **Source**: `https://data.chain.link/streams/btc-usd`
- **Table**: `ohlcv_raw` with `source='chainlink'`

**VOLUME (Features Only)**: Binance BTCUSDT (Optional)
- **Used for**: Volume-based indicators (OBV, VPT, VWAP)
- **Why**: Chainlink doesn't provide volume data
- **Source**: Binance `/api/v3/klines` endpoint
- **Table**: `ohlcv_raw` with `source='binance'` (volume column only)

### Data Flow

```
Chainlink API (every 60s)
    ↓
Price → ohlcv_raw (source='chainlink')
    ↓
Feature Engineer → Technical Indicators
    ↓
Inference → Predictions (based on Chainlink price)
    ↓
Polymarket-compatible predictions
```

**Never use Binance/other exchanges as primary price source** - they won't match Polymarket settlement!

## Quick Start

### 1. Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA 12.8 support
- 16GB+ RAM
- 50GB+ disk space

### 2. Environment Setup

```bash
# Clone repository
git clone <repo-url>
cd btc-ml-system

# Create environment file
cp .env.example .env

# Edit .env with your credentials
nano .env
```

### 3. Build and Start Services

**IMPORTANT: Choose isolation level**

**Option 1: FULL ISOLATION (Recommended for Production)**
```bash
# No ports exposed to host - maximum security
docker-compose -f docker-compose.isolated.yml build
docker-compose -f docker-compose.isolated.yml up -d

# Access database only via docker exec:
docker exec btc-ml-db psql -U mluser -d btc_ml
```

**Option 2: WITH DATABASE PORT (Development/Monitoring)**
```bash
# Exposes PostgreSQL port (configurable via DB_EXTERNAL_PORT in .env)
docker-compose build
docker-compose up -d

# Access database from host:
psql -h localhost -p 5432 -U mluser -d btc_ml
# Or custom port if DB_EXTERNAL_PORT=15432
psql -h localhost -p 15432 -U mluser -d btc_ml
```

**Verify services are running:**
```bash
docker-compose ps
```

### 4. Verify Database Setup

```bash
# Check database connectivity
docker exec btc-ml-db psql -U mluser -d btc_ml -c "SELECT version();"

# Check tables created
docker exec btc-ml-db psql -U mluser -d btc_ml -c "\dt"
```

### 5. Monitor Services

```bash
# Watch Chainlink poller logs
docker logs -f btc-ml-chainlink

# Monitor feature engineer
docker logs -f btc-ml-features

# View inference predictions
docker logs -f btc-ml-inference

# Tail prediction logs
tail -f logs/inference.log
```

## Service Details

### Chainlink Poller
- **Purpose**: Fetches BTC/USD price from Chainlink every 60 seconds
- **Database**: Writes to `ohlcv_raw` table
- **Health**: Logs every successful fetch with timestamp

### Feature Engineer
- **Purpose**: Calculates 15+ technical indicators (RSI, MACD, ATR, etc.)
- **Trigger**: Runs every 60 seconds after new price data
- **Output**: Writes to `technical_indicators` table

### Training Service
- **Purpose**: Daily model retraining using walk-forward validation
- **Schedule**: Runs at 02:00 UTC daily (APScheduler)
- **Models**: LSTM (PyTorch) + XGBoost ensemble
- **Output**: Model artifacts saved to `/models/active/`

### Inference Service
- **Purpose**: Real-time price direction predictions
- **Frequency**: Every 60 seconds
- **Latency**: <5 seconds target
- **Output**: Logs predictions in CSV format + writes to `predictions` table

## Database Schema

### Tables

1. **ohlcv_raw** - Raw 1-minute price data from Chainlink
2. **ohlcv_15m** - Aggregated 15-minute candles
3. **technical_indicators** - Calculated indicators (RSI, MACD, etc.)
4. **predictions** - Model inference outputs
5. **training_metadata** - Model training history
6. **model_registry** - Active/archived model tracking

## Model Performance

### Expected Metrics
- **Accuracy**: 51-55% (target >52%)
- **Win Rate**: 51-53%
- **Latency**: <5 seconds per inference
- **Uptime**: 99.9%

### Architecture
- **LSTM**: Temporal feature extraction (32 latent features)
- **XGBoost**: Non-linear relationship modeling
- **Ensemble**: Voting from Model A, B, C

## Development Workflow

### Testing Individual Services

```bash
# Test database connection
python scripts/test_db.py

# Run feature calculation test
docker exec btc-ml-features python -c "from indicators import TechnicalIndicatorCalculator; print('OK')"

# Test inference pipeline
python scripts/test_inference.py
```

### Manual Model Training

```bash
# Trigger training manually
docker exec btc-ml-training python /app/train.py --full-retrain

# Check training logs
docker logs btc-ml-training -f
```

### Monitoring GPU Usage

```bash
# Check GPU utilization
docker exec btc-ml-inference nvidia-smi

# Monitor GPU memory
watch -n 1 nvidia-smi
```

## Troubleshooting

### Service Won't Start
```bash
# Check logs
docker logs <service-name>

# Restart specific service
docker-compose restart <service-name>

# Rebuild if code changed
docker-compose up -d --build <service-name>
```

### Database Connection Issues
```bash
# Verify TimescaleDB is healthy
docker exec btc-ml-db pg_isready -U mluser

# Check database logs
docker logs btc-ml-db

# Reset database (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d timescaledb
```

### Inference Latency High (>5s)
```bash
# Profile GPU usage
docker exec btc-ml-inference nvidia-smi

# Check database query times
docker exec btc-ml-db psql -U mluser -d btc_ml -c "EXPLAIN ANALYZE SELECT * FROM technical_indicators ORDER BY time DESC LIMIT 60;"

# View detailed timing logs
docker logs btc-ml-inference | grep "took"
```

### Missing Predictions
```bash
# Check if feature calculation is running
docker logs btc-ml-features --tail 50

# Verify Chainlink poller is fetching
docker logs btc-ml-chainlink --tail 50

# Check database has recent data
docker exec btc-ml-db psql -U mluser -d btc_ml -c "SELECT COUNT(*) FROM ohlcv_raw WHERE time > NOW() - INTERVAL '1 hour';"
```

## Directory Structure

```
btc-ml-system/
├── docker-compose.yml          # Orchestration config
├── .env.example                # Environment template
├── init.sql                    # Database schema
├── README.md                   # This file
│
├── services/
│   ├── chainlink-poller/       # Price data ingestion
│   ├── feature-engineer/       # Indicator calculation
│   ├── training/               # Model training
│   ├── inference/              # Real-time predictions
│   └── shared/                 # Common utilities
│
├── models/
│   ├── active/                 # Production models
│   ├── archive/                # Historical models
│   └── registry.json           # Model tracking
│
├── data/
│   └── init.sql                # Database initialization
│
├── logs/                       # Application logs
│
└── scripts/                    # Utility scripts
    ├── start.sh                # Start all services
    ├── test_db.py              # Database connectivity test
    └── backtest_sample.py      # Backtesting utility
```

## API & Integration

### Prediction Output Format

```csv
timestamp,direction,confidence,model_name
2026-01-06T11:07:00Z,UP,0.63,ModelA
2026-01-06T11:08:00Z,DOWN,0.51,ModelA
2026-01-06T11:09:00Z,UP,0.58,Ensemble
```

### Database Queries

```sql
-- Get last hour predictions
SELECT time, direction, confidence, model_name 
FROM predictions 
WHERE time > NOW() - INTERVAL '1 hour'
ORDER BY time DESC;

-- Check prediction accuracy
SELECT 
    direction,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence
FROM predictions 
WHERE time > NOW() - INTERVAL '24 hours'
GROUP BY direction;

-- Model performance metrics
SELECT * FROM training_metadata 
ORDER BY trained_at DESC 
LIMIT 5;
```

## Maintenance

### Daily Operations
- Monitor logs for errors: `docker-compose logs -f`
- Check GPU utilization: `nvidia-smi`
- Verify predictions are being generated: `tail -f logs/inference.log`

### Weekly Operations
- Review model accuracy: Query `predictions` table
- Check disk space: `df -h`
- Backup database: `pg_dump -U mluser btc_ml > backup.sql`

### Monthly Operations
- Archive old models: Move from `models/active/` to `models/archive/`
- Clean old logs: `find logs/ -mtime +30 -delete`
- Review training metrics: Query `training_metadata` table

## Security Notes

### Network Isolation

The system provides two deployment modes:

**1. Fully Isolated Mode** (`docker-compose.isolated.yml`)
- ✅ **NO ports exposed** to host machine
- ✅ All services communicate via internal `ml-network` only
- ✅ Database accessible only via `docker exec`
- ✅ **Recommended for production**
- ✅ Use: `docker-compose -f docker-compose.isolated.yml up -d`

**2. Development Mode** (`docker-compose.yml`)
- ⚠️ Database port exposed (configurable via `DB_EXTERNAL_PORT`)
- ✅ Useful for monitoring, debugging, external tools
- ✅ Port can be changed to avoid conflicts
- ⚠️ Set `DB_EXTERNAL_PORT=""` in `.env` to disable exposure

### Additional Security

- Change default passwords in `.env`
- Restrict database port exposure (5432) in production → **use isolated mode**
- Use secrets management for production deployment
- Enable SSL/TLS for database connections (see PostgreSQL docs)
- Implement rate limiting on external API calls
- Regular security updates: `docker-compose pull && docker-compose up -d`

### Firewall Rules (if using exposed ports)

```bash
# Allow only from specific IPs
iptables -A INPUT -p tcp --dport 5432 -s YOUR_IP -j ACCEPT
iptables -A INPUT -p tcp --dport 5432 -j DROP
```

## Performance Tuning

### GPU Optimization
- Adjust batch sizes in training config
- Enable mixed precision training (FP16)
- Use CUDA streams for parallel inference

### Database Optimization
- Regular VACUUM and ANALYZE
- Adjust TimescaleDB chunk intervals
- Monitor index usage

### Service Optimization
- Tune APScheduler thread pool size
- Adjust connection pool sizes
- Enable query caching where appropriate

## Contributing

See individual service READMEs for development guidelines:
- `services/chainlink-poller/README.md`
- `services/feature-engineer/README.md`
- `services/training/README.md`
- `services/inference/README.md`

## License

Proprietary - All Rights Reserved

## Support

For issues and questions:
1. Check logs first: `docker-compose logs <service>`
2. Review troubleshooting section above
3. Contact system administrator

---

**Version**: 1.0.0  
**Last Updated**: January 2026  
**Status**: Production Ready