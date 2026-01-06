"""
Chainlink BTC/USD Price Poller Service
Fetches latest Bitcoin price from Chainlink Data Streams every minute
Stores in TimescaleDB for downstream processing
"""

import os
import sys
import time
import signal
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any

import requests
from apscheduler.schedulers.background import BackgroundScheduler

# Add parent directory to path for shared modules


from db_client import create_db_client_from_env
from logger import setup_logger_from_env
import constants

# Setup logger
logger = setup_logger_from_env('chainlink-poller')


class ChainlinkPoller:
    """
    Polls Chainlink BTC/USD price feed and stores to database.
    
    Features:
    - Automatic retry logic on failures
    - Connection pooling for database efficiency
    - Structured logging
    - Graceful shutdown handling
    - ON CONFLICT handling for idempotency
    """
    
    def __init__(
        self,
        api_url: str = constants.CHAINLINK_API_URL,
        fetch_interval: int = constants.CHAINLINK_FETCH_INTERVAL,
        max_retries: int = constants.CHAINLINK_MAX_RETRIES,
        retry_delay: int = constants.CHAINLINK_RETRY_DELAY
    ):
        """
        Initialize Chainlink poller.
        
        Args:
            api_url: Chainlink API endpoint
            fetch_interval: Seconds between fetches
            max_retries: Maximum retry attempts per fetch
            retry_delay: Seconds to wait between retries
        """
        self.api_url = api_url
        self.fetch_interval = fetch_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize database client
        try:
            self.db = create_db_client_from_env()
            logger.info("Database client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database client: {e}")
            raise
        
        # Initialize scheduler
        self.scheduler = BackgroundScheduler()
        
        # Statistics
        self.total_fetches = 0
        self.successful_fetches = 0
        self.failed_fetches = 0
        
        # Shutdown flag
        self.shutdown_requested = False
    
    def fetch_price(self) -> Optional[Dict[str, Any]]:
        """
        Fetch latest BTC/USD price from Chainlink.
        
        Returns:
            Dictionary with price data, or None if failed
            
        Example return:
            {
                'price': 42157.50,
                'timestamp': datetime(...),
                'source': 'chainlink'
            }
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    self.api_url,
                    timeout=constants.API_REQUEST_TIMEOUT
                )
                response.raise_for_status()
                
                # Parse response
                # Note: Actual Chainlink API response format may vary
                # Adjust parsing logic based on actual response structure
                data = response.json()
                
                # Extract price (adjust based on actual API structure)
                # Common formats:
                # 1. {"price": "42157.50", "timestamp": "..."}
                # 2. {"data": {"price": ..., "timestamp": ...}}
                # 3. {"result": ...}
                
                if isinstance(data, dict):
                    # Try common field names
                    price_value = None
                    timestamp_value = None
                    
                    # Try direct fields
                    if 'price' in data:
                        price_value = float(data['price'])
                    elif 'result' in data and isinstance(data['result'], dict):
                        price_value = float(data['result'].get('price', 0))
                    elif 'data' in data and isinstance(data['data'], dict):
                        price_value = float(data['data'].get('price', 0))
                    
                    # Get timestamp
                    if 'timestamp' in data:
                        timestamp_value = data['timestamp']
                    elif 'updatedAt' in data:
                        timestamp_value = data['updatedAt']
                    else:
                        timestamp_value = datetime.utcnow().isoformat()
                    
                    if price_value and price_value > 0:
                        logger.info(
                            f"✓ Chainlink GROUND TRUTH: BTC/USD = ${price_value:.2f} (Polymarket Settlement Source)",
                            extra={'extra_data': {
                                'price': price_value,
                                'source': 'chainlink',
                                'ground_truth': True,
                                'polymarket_settlement_source': True,
                                'attempt': attempt + 1
                            }}
                        )
                        
                        return {
                            'price': price_value,
                            'timestamp': timestamp_value,
                            'source': 'chainlink'
                        }
                
                # If we couldn't parse, log and retry
                logger.warning(f"Unable to parse price from response: {data}")
                
            except requests.exceptions.Timeout:
                logger.warning(
                    f"Fetch attempt {attempt + 1} timed out",
                    extra={'extra_data': {'attempt': attempt + 1}}
                )
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Fetch attempt {attempt + 1} failed: {e}",
                    extra={'extra_data': {'attempt': attempt + 1, 'error': str(e)}}
                )
            except (ValueError, KeyError) as e:
                logger.warning(
                    f"Failed to parse response on attempt {attempt + 1}: {e}",
                    extra={'extra_data': {'attempt': attempt + 1, 'error': str(e)}}
                )
            
            # Wait before retry (except on last attempt)
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        
        # All retries failed
        logger.error(f"Failed to fetch price after {self.max_retries} attempts")
        return None
    
    def store_price(self, price_data: Dict[str, Any]) -> bool:
        """
        Store price data in TimescaleDB.
        
        Args:
            price_data: Dictionary containing price, timestamp, source
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not price_data:
            return False
        
        try:
            # Parse timestamp (handle various formats)
            timestamp = price_data['timestamp']
            if isinstance(timestamp, str):
                # Try ISO format first
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    # Fallback to current time
                    timestamp = datetime.utcnow()
            elif not isinstance(timestamp, datetime):
                timestamp = datetime.utcnow()
            
            price = Decimal(str(price_data['price']))
            
            # Insert into database
            # For instant data (no OHLCV aggregation), use close price for all fields
            data = {
                'time': timestamp,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 0,  # Chainlink doesn't provide volume
                'source': price_data['source']
            }
            
            self.db.insert(
                constants.TABLE_OHLCV_RAW,
                data,
                on_conflict="(time) DO UPDATE SET close = EXCLUDED.close, volume = EXCLUDED.volume"
            )
            
            logger.info(
                f"✓ Stored Chainlink GROUND TRUTH: ${price:.2f} at {timestamp}",
                extra={'extra_data': {
                    'price': float(price),
                    'timestamp': timestamp.isoformat(),
                    'source': 'chainlink',
                    'ground_truth': True
                }}
            )
            
            return True
            
        except Exception as e:
            logger.error(
                f"Failed to store price: {e}",
                extra={'extra_data': {'error': str(e), 'price_data': price_data}}
            )
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check service health.
        
        Returns:
            Dictionary with health status
        """
        db_healthy = self.db.health_check()
        
        # Try to fetch latest price from DB
        last_price = None
        try:
            result = self.db.fetch_one(
                f"SELECT time, close FROM {constants.TABLE_OHLCV_RAW} ORDER BY time DESC LIMIT 1"
            )
            if result:
                last_price = {
                    'time': result['time'],
                    'price': float(result['close'])
                }
        except:
            pass
        
        health_status = {
            'service': 'chainlink-poller',
            'database_connected': db_healthy,
            'total_fetches': self.total_fetches,
            'successful_fetches': self.successful_fetches,
            'failed_fetches': self.failed_fetches,
            'last_price': last_price,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store health check in database
        try:
            self.db.insert(
                constants.TABLE_SYSTEM_HEALTH,
                {
                    'check_time': datetime.utcnow(),
                    'service_name': 'chainlink-poller',
                    'status': 'healthy' if db_healthy else 'down',
                    'latency_ms': 0,
                    'additional_info': health_status
                },
                on_conflict="DO NOTHING"
            )
        except:
            pass
        
        return health_status
    
    def fetch_and_store(self):
        """
        Main job: fetch price and store to database.
        Called by scheduler every fetch_interval seconds.
        """
        if self.shutdown_requested:
            return
        
        self.total_fetches += 1
        
        # Fetch price
        price_data = self.fetch_price()
        
        # Store if successful
        if price_data:
            success = self.store_price(price_data)
            if success:
                self.successful_fetches += 1
            else:
                self.failed_fetches += 1
        else:
            self.failed_fetches += 1
        
        # Periodic health check (every 10 fetches)
        if self.total_fetches % 10 == 0:
            health = self.health_check()
            logger.info(
                f"Health check: {health['successful_fetches']}/{health['total_fetches']} successful",
                extra={'extra_data': health}
            )
    
    def start(self):
        """
        Start the polling service.
        Runs indefinitely until shutdown requested.
        """
        logger.info(
            f"Starting Chainlink Poller - GROUND TRUTH SOURCE for Polymarket (interval: {self.fetch_interval}s)",
            extra={'extra_data': {
                'api_url': self.api_url,
                'fetch_interval': self.fetch_interval,
                'max_retries': self.max_retries,
                'role': 'ground_truth_price_source',
                'polymarket_settlement': True
            }}
        )
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Schedule periodic fetch job
        self.scheduler.add_job(
            self.fetch_and_store,
            'interval',
            seconds=self.fetch_interval,
            id='fetch_job',
            max_instances=1
        )
        
        # Start scheduler
        self.scheduler.start()
        
        # Run initial fetch immediately
        self.fetch_and_store()
        
        # Keep service running
        try:
            while not self.shutdown_requested:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.shutdown()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown")
        self.shutdown_requested = True
    
    def shutdown(self):
        """
        Gracefully shutdown the service.
        """
        logger.info("Shutting down Chainlink poller...")
        
        # Stop scheduler
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            logger.info("Scheduler stopped")
        
        # Close database connections
        if hasattr(self, 'db'):
            self.db.close()
            logger.info("Database connections closed")
        
        # Final statistics
        logger.info(
            f"Final statistics: {self.successful_fetches}/{self.total_fetches} successful fetches",
            extra={'extra_data': {
                'total': self.total_fetches,
                'successful': self.successful_fetches,
                'failed': self.failed_fetches
            }}
        )
        
        logger.info("Chainlink poller shutdown complete")


def main():
    """Entry point for Chainlink poller service."""
    logger.info("=" * 80)
    logger.info("Chainlink BTC/USD Poller - GROUND TRUTH SOURCE for Polymarket")
    logger.info("=" * 80)
    logger.info("NOTE: This service provides the official price used for Polymarket settlement")
    logger.info("      Do NOT use Binance or other sources as primary price - they are for features only")
    logger.info("=" * 80)
    
    try:
        # Create and start poller
        poller = ChainlinkPoller()
        poller.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
