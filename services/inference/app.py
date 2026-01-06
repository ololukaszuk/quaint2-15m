"""
Inference Service
Real-time price direction predictions using trained ML models
"""

import os
import sys
import time
import signal
from datetime import datetime
from typing import Optional

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.db_client import create_db_client_from_env
from shared.logger import setup_logger_from_env
from shared import constants
from predictor import Predictor, prepare_inference_features

logger = setup_logger_from_env('inference')


class InferenceService:
    """
    Real-time prediction service.
    """
    
    def __init__(
        self,
        model_path: str = constants.MODEL_ACTIVE_PATH,
        inference_interval: int = constants.INFERENCE_INTERVAL
    ):
        """
        Initialize inference service.
        
        Args:
            model_path: Path to model directory
            inference_interval: Seconds between inferences
        """
        self.model_path = model_path
        self.inference_interval = inference_interval
        
        # Initialize components
        self.db = create_db_client_from_env()
        self.predictor = Predictor(model_path)
        
        self.scheduler = BackgroundScheduler()
        self.shutdown_requested = False
        
        # Statistics
        self.total_inferences = 0
        self.successful_inferences = 0
        
        logger.info("Inference Service initialized")
    
    def get_latest_features(self, lookback_minutes: int = 60) -> Optional[pd.DataFrame]:
        """
        Get latest OHLCV + indicators for inference.
        
        CRITICAL: Uses Chainlink prices ONLY (ground truth for Polymarket settlement)
        
        Args:
            lookback_minutes: Minutes of history to fetch
            
        Returns:
            DataFrame with features, or None if insufficient
        """
        try:
            query = f"""
                SELECT 
                    o.time, o.open, o.high, o.low, o.close, o.volume, o.source,
                    t.rsi_14, t.macd, t.macd_hist, t.atr_14, t.stoch_k,
                    t.ema_12, t.ema_26, t.volatility_20, t.bb_width
                FROM {constants.TABLE_OHLCV_RAW} o
                LEFT JOIN {constants.TABLE_TECHNICAL_INDICATORS} t ON o.time = t.time
                WHERE o.time > NOW() - INTERVAL '{lookback_minutes} minutes'
                    AND o.source = 'chainlink'
                ORDER BY o.time ASC
            """
            
            df = pd.DataFrame(self.db.fetch_all(query))
            
            if df.empty or len(df) < constants.MIN_CANDLES_FOR_INFERENCE:
                logger.warning(f"Insufficient Chainlink data: {len(df)} rows")
                return None
            
            logger.info(
                f"Fetched {len(df)} feature rows from Chainlink (ground truth)",
                extra={'extra_data': {
                    'source': 'chainlink',
                    'ground_truth': True,
                    'rows': len(df),
                    'latest_price': float(df['close'].iloc[-1]) if len(df) > 0 else None
                }}
            )
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch features: {e}")
            return None
    
    def run_inference(self):
        """Execute one inference cycle."""
        if self.shutdown_requested:
            return
        
        self.total_inferences += 1
        start_time = time.time()
        
        try:
            # Fetch features
            df = self.get_latest_features()
            
            if df is None:
                return
            
            # Prepare features
            combined_features, lstm_features = prepare_inference_features(df, self.predictor)
            
            # Get latest Chainlink price for logging
            latest_chainlink_price = float(df['close'].iloc[-1])
            
            # Generate prediction
            direction, confidence = self.predictor.predict(combined_features)
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Log prediction (CSV format for monitoring)
            timestamp = datetime.utcnow().isoformat() + 'Z'
            log_line = f"{timestamp},{direction},{confidence:.4f},{constants.MODEL_A},chainlink_price=${latest_chainlink_price:.2f}"
            logger.info(
                f"PREDICTION: {direction} (confidence={confidence:.4f}) based on Chainlink price ${latest_chainlink_price:.2f}",
                extra={'extra_data': {
                    'direction': direction,
                    'confidence': confidence,
                    'chainlink_price': latest_chainlink_price,
                    'ground_truth_source': 'chainlink',
                    'polymarket_compatible': True,
                    'latency_ms': latency_ms
                }}
            )
            
            # Store in database
            self.store_prediction(timestamp, direction, confidence, lstm_features, latency_ms)
            
            self.successful_inferences += 1
            
            # Warning if latency high
            if latency_ms > constants.INFERENCE_LATENCY_WARNING_THRESHOLD * 1000:
                logger.warning(
                    f"High inference latency: {latency_ms}ms",
                    extra={'extra_data': {'latency_ms': latency_ms}}
                )
            
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
    
    def store_prediction(
        self,
        timestamp: str,
        direction: str,
        confidence: float,
        lstm_latent: list,
        latency_ms: int
    ):
        """
        Store prediction in database.
        
        Args:
            timestamp: ISO format timestamp
            direction: UP or DOWN
            confidence: Confidence score
            lstm_latent: LSTM latent features
            latency_ms: Inference latency in milliseconds
        """
        try:
            data = {
                'time': timestamp,
                'direction': direction,
                'confidence': confidence,
                'model_name': constants.MODEL_A,
                'lstm_latent': lstm_latent.tolist() if hasattr(lstm_latent, 'tolist') else list(lstm_latent),
                'latency_ms': latency_ms
            }
            
            self.db.insert(
                constants.TABLE_PREDICTIONS,
                data,
                on_conflict="DO NOTHING"
            )
            
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
    
    def start(self):
        """Start the inference service."""
        logger.info(f"Starting Inference Service (interval: {self.inference_interval}s)")
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Schedule inference job
        self.scheduler.add_job(
            self.run_inference,
            'interval',
            seconds=self.inference_interval,
            id='inference_job',
            max_instances=1
        )
        
        self.scheduler.start()
        
        # Initial inference
        self.run_inference()
        
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
        logger.info("Shutting down Inference Service...")
        
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
        
        if hasattr(self, 'db'):
            self.db.close()
        
        logger.info(f"Final: {self.successful_inferences}/{self.total_inferences} successful inferences")
        logger.info("Inference Service shutdown complete")


def main():
    """Entry point."""
    logger.info("=" * 80)
    logger.info("Inference Service Starting")
    logger.info("=" * 80)
    logger.info("CRITICAL: Predictions based on Chainlink BTC/USD (Polymarket settlement source)")
    logger.info("          Volume indicators may use Binance data for features only")
    logger.info("=" * 80)
    
    try:
        service = InferenceService()
        service.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
