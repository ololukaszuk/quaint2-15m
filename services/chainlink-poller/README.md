# Chainlink Poller Service

Fetches BTC/USD price from Chainlink Data Streams every 60 seconds and stores in TimescaleDB.

## Features

- Automatic retry on failures (3 retries with 5s delay)
- Connection pooling for database efficiency
- Structured logging with JSON support
- Health checks
- Graceful shutdown

## Configuration

Environment variables:
- `CHAINLINK_API_URL`: Chainlink endpoint (default: https://data.chain.link/streams/btc-usd)
- `CHAINLINK_FETCH_INTERVAL`: Seconds between fetches (default: 60)
- `CHAINLINK_MAX_RETRIES`: Max retry attempts (default: 3)

## Output

Writes to `ohlcv_raw` table:
- `time`: Timestamp
- `open`, `high`, `low`, `close`: Price (all same for instant data)
- `volume`: 0 (not available from Chainlink)
- `source`: 'chainlink'

## Monitoring

View logs:
```bash
docker logs -f btc-ml-chainlink
```

Check database:
```sql
SELECT * FROM ohlcv_raw ORDER BY time DESC LIMIT 10;
```
