# Training Service - Full Production

Daily model retraining with LSTM + XGBoost on Chainlink data.

## Architecture

1. **Data Loading**: Fetches 12 months of Chainlink OHLCV (ground truth)
2. **Label Creation**: Binary labels (UP/DOWN) for 15-min ahead prediction
3. **LSTM Training**: Extracts 32 temporal features from 60-min sequences
4. **XGBoost Training**: Combines LSTM features + technical indicators
5. **Model Deployment**: Saves to `/models/active/` for inference service

## Schedule

- Runs daily at 02:00 UTC (configurable via `TRAINING_SCHEDULE_CRON`)
- Also runs immediately on startup

## Performance Targets

- Accuracy: >52% on validation set
- AUC-ROC: >0.55
- Training time: <30 minutes (with GPU)

## Models Saved

- `lstm_model.pt`: PyTorch LSTM state dict
- `xgb_model.pkl`: XGBoost Booster object

## Monitoring

```bash
docker logs -f btc-ml-training
```

## Manual Training Trigger

```bash
docker exec btc-ml-training python /app/train.py
```
