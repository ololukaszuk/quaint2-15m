# Feature Engineer Service

Calculates technical indicators from raw OHLCV data every 60 seconds.

## Features

- 15+ technical indicators (RSI, MACD, ATR, Bollinger Bands, etc.)
- Vectorized calculation using NumPy/Pandas
- Z-score normalization
- Efficient batch processing

## Indicators Calculated

**Momentum**: RSI(14), MACD, Stochastic
**Volatility**: ATR(14), Bollinger Bands, Rolling Volatility
**Trend**: EMA(12, 26), SMA(20)
**Volume**: OBV, VPT, Volume SMA

## Output

Writes to `technical_indicators` table with all calculated indicators.

## Monitoring

```bash
docker logs -f btc-ml-features
```
