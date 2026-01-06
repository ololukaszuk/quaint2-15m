"""
Feature Engineer Service
Calculates technical indicators from raw OHLCV data
Stores results in TimescaleDB for ML inference
"""

import os
import sys
import time
import signal
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.db_client import create_db_client_from_env
from shared.logger import setup_logger_from_env
from shared import constants
from indicators import TechnicalIndicatorCalculator, FeatureNormalizer

logger = setup_logger_from_env('feature-engineer')


class FeatureEngineer:
    """
    Feature engineering service that calculates technical indicators.
    """
    
    def __init__(
        self,
        calculation_interval: int = constants.FEATURE_CALCULATION_INTERVAL,
        lookback_minutes: int = constants.FEATURE_LOOKBACK_MINUTES
    ):
        """
        Initialize feature engineer.
        
        Args:
            calculation_interval: Seconds between calculations
            lookback_minutes: Minutes of historical data to process
        """
        self.calculation_interval = calculation_interval
        self.lookback_minutes = lookback_minutes
        
        # Initialize components
        self.db = create_db_client_from_env()
        self.calculator = TechnicalIndicatorCalculator()
        self.normalizer = FeatureNormalizer()
        
        self.scheduler = BackgroundScheduler()
        self.shutdown_requested = False
        
        # Statistics
        self.total_calculations = 0
        self.successful_calculations = 0
        
        logger.info("Feature Engineer initialized")
    
    def get_latest_ohlcv(self, lookback_minutes: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch latest OHLCV data from database.
        
        NOTE: Price data comes from Chainlink (ground truth for Polymarket)
              Volume data may come from Binance (for technical indicators)
        
        Args:
            lookback_minutes: Minutes of historical data
            
        Returns:
            DataFrame with OHLCV data, or None if insufficient
        """
        try:
            query = f"""
                SELECT time, open, high, low, close, volume, source
                FROM {constants.TABLE_OHLCV_RAW}
                WHERE time > NOW() - INTERVAL '{lookback_minutes} minutes'
                    AND source = 'chainlink'
                ORDER BY time ASC
            """
            
            df = pd.DataFrame(self.db.fetch_all(query))
            
            if df.empty or len(df) < 30:
                logger.warning(f"Insufficient Chainlink data: {len(df)} rows (need >= 30)")
                return None
            
            logger.info(
                f"Fetched {len(df)} OHLCV rows from Chainlink (ground truth source)",
                extra={'extra_data': {
                    'source': 'chainlink',
                    'ground_truth': True,
                    'rows': len(df)
                }}
            )
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        try:
            df_with_indicators = self.calculator.calculate_all(df)
            logger.info(f"Calculated indicators for {len(df_with_indicators)} rows")
            return df_with_indicators
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            raise
    
    def store_indicators(self, df: pd.DataFrame) -> bool:
        """
        Store indicators in database.
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            True if stored successfully
        """
        try:
            # Select indicator columns
            indicator_columns = [
                'time', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'atr_14', 'stoch_k', 'stoch_d', 'ema_12', 'ema_26',
                'sma_20', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'volatility_20', 'obv', 'vpt', 'volume_sma'
            ]
            
            # Filter available columns
            available_cols = [col for col in indicator_columns if col in df.columns]
            df_indicators = df[available_cols].copy()
            
            # Remove rows with all NaN indicators (first rows)
            df_indicators = df_indicators.dropna(subset=[col for col in available_cols if col != 'time'], how='all')
            
            if df_indicators.empty:
                logger.warning("No indicators to store (all NaN)")
                return False
            
            # Convert to records for insertion
            records = df_indicators.to_dict('records')
            
            # Bulk insert
            self.db.bulk_insert(
                constants.TABLE_TECHNICAL_INDICATORS,
                records,
                on_conflict="(time) DO UPDATE SET rsi_14 = EXCLUDED.rsi_14, macd = EXCLUDED.macd, atr_14 = EXCLUDED.atr_14"
            )
            
            logger.info(f"Stored {len(records)} indicator rows")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store indicators: {e}")
            return False
    
    def calculate_and_store(self):
        """Main job: calculate indicators and store."""
        if self.shutdown_requested:
            return
        
        self.total_calculations += 1
        
        try:
            # Fetch OHLCV
            df = self.get_latest_ohlcv(self.lookback_minutes)
            
            if df is None:
                return
            
            # Calculate indicators
            df_with_indicators = self.calculate_indicators(df)
            
            # Store indicators
            success = self.store_indicators(df_with_indicators)
            
            if success:
                self.successful_calculations += 1
                logger.info(f"Feature calculation cycle complete ({self.successful_calculations}/{self.total_calculations})")
            
        except Exception as e:
            logger.error(f"Feature calculation failed: {e}")
    
    def start(self):
        """Start the feature engineer service."""
        logger.info(f"Starting Feature Engineer (interval: {self.calculation_interval}s)")
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Schedule job
        self.scheduler.add_job(
            self.calculate_and_store,
            'interval',
            seconds=self.calculation_interval,
            id='calculation_job',
            max_instances=1
        )
        
        self.scheduler.start()
        
        # Initial calculation
        self.calculate_and_store()
        
        # Keep running
        try:
            while not self.shutdown_requested:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.shutdown()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}")
        self.shutdown_requested = True
    
    def shutdown(self):
        """Gracefully shutdown."""
        logger.info("Shutting down Feature Engineer...")
        
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
        
        if hasattr(self, 'db'):
            self.db.close()
        
        logger.info("Feature Engineer shutdown complete")


def main():
    """Entry point."""
    logger.info("=" * 80)
    logger.info("Feature Engineer Service Starting")
    logger.info("=" * 80)
    logger.info("DATA SOURCES:")
    logger.info("  - PRICE: Chainlink BTC/USD (ground truth for Polymarket)")
    logger.info("  - VOLUME: Binance BTCUSDT (for indicators only)")
    logger.info("=" * 80)
    
    try:
        engineer = FeatureEngineer()
        engineer.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
