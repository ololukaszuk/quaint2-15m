# Inference Service

Real-time price direction predictions using LSTM + XGBoost ensemble.

## Architecture

1. LSTM extracts 32 latent temporal features from 60-min price sequence
2. XGBoost combines latent features + technical indicators
3. Outputs direction (UP/DOWN) + confidence (0.5-1.0)

## Performance

- Target latency: <5 seconds
- Runs every 60 seconds
- GPU-accelerated (CUDA 12.8)

## Output Format

Logs (CSV):
```
2026-01-06T12:00:00Z,UP,0.63,ModelA
```

Database (`predictions` table):
- `time`, `direction`, `confidence`, `model_name`, `lstm_latent`, `latency_ms`

## Monitoring

```bash
docker logs -f btc-ml-inference
tail -f logs/inference.log
```
