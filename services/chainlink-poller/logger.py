"""
Shared Structured Logging Module
JSON-formatted logging with rotation support
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Any, Dict
from logging.handlers import RotatingFileHandler


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs JSON-structured logs.
    
    Format:
    {
        "timestamp": "2026-01-06T12:00:00.000Z",
        "level": "INFO",
        "service": "chainlink-poller",
        "message": "Fetched BTC/USD: 42157.50",
        "extra": {...}
    }
    """
    
    def __init__(self, service_name: str = "unknown"):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "level": record.levelname,
            "service": self.service_name,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        if hasattr(record, 'extra_data'):
            log_data["extra"] = record.extra_data
        
        return json.dumps(log_data)


class SimpleFormatter(logging.Formatter):
    """
    Simple human-readable formatter for console output.
    
    Format: 2026-01-06T12:00:00Z - INFO - chainlink-poller - Message here
    """
    
    def __init__(self, service_name: str = "unknown"):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as simple text."""
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        return f"{timestamp} - {record.levelname} - {self.service_name} - {record.getMessage()}"


def setup_logger(
    service_name: str,
    log_level: str = "INFO",
    log_dir: str = "/app/logs",
    log_to_file: bool = True,
    log_to_console: bool = True,
    max_bytes: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 10,
    json_format: bool = False
) -> logging.Logger:
    """
    Configure and return a logger with file and console handlers.
    
    Args:
        service_name: Name of the service (used in log messages)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        log_to_file: Enable file logging
        log_to_console: Enable console logging
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep
        json_format: Use JSON format for file logs (simple format for console)
    
    Returns:
        Configured logger instance
    
    Example:
        logger = setup_logger("chainlink-poller", log_level="INFO")
        logger.info("Service started")
        logger.error("Connection failed", extra={'extra_data': {'retry': 3}})
    """
    # Create logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler with rotation
    if log_to_file:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"{service_name}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Use JSON format for files
        if json_format:
            file_formatter = StructuredFormatter(service_name)
        else:
            file_formatter = SimpleFormatter(service_name)
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Use simple format for console (easier to read)
        console_formatter = SimpleFormatter(service_name)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def setup_logger_from_env(service_name: str) -> logging.Logger:
    """
    Setup logger using environment variables for configuration.
    
    Environment Variables:
        LOG_LEVEL: Logging level (default: INFO)
        LOG_DIR: Log directory (default: /app/logs)
        LOG_TO_FILE: Enable file logging (default: true)
        LOG_TO_CONSOLE: Enable console logging (default: true)
        LOG_MAX_SIZE_MB: Max log file size in MB (default: 100)
        LOG_MAX_FILES: Number of backup files (default: 10)
        LOG_JSON_FORMAT: Use JSON format (default: false)
    
    Args:
        service_name: Name of the service
    
    Returns:
        Configured logger instance
    """
    return setup_logger(
        service_name=service_name,
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        log_dir=os.getenv('LOG_DIR', '/app/logs'),
        log_to_file=os.getenv('LOG_TO_FILE', 'true').lower() == 'true',
        log_to_console=os.getenv('LOG_TO_CONSOLE', 'true').lower() == 'true',
        max_bytes=int(os.getenv('LOG_MAX_SIZE_MB', '100')) * 1024 * 1024,
        backup_count=int(os.getenv('LOG_MAX_FILES', '10')),
        json_format=os.getenv('LOG_JSON_FORMAT', 'false').lower() == 'true'
    )


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context to all log messages.
    
    Example:
        logger = setup_logger("my-service")
        context_logger = LoggerAdapter(logger, {"request_id": "123"})
        context_logger.info("Processing request")
        # Output includes request_id in extra data
    """
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add context to log record."""
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        # Merge adapter context with log call extra data
        if 'extra_data' not in kwargs['extra']:
            kwargs['extra']['extra_data'] = {}
        
        kwargs['extra']['extra_data'].update(self.extra)
        
        return msg, kwargs


# Example usage and testing
if __name__ == "__main__":
    # Test logger setup
    logger = setup_logger(
        "test-service",
        log_level="DEBUG",
        log_dir="/tmp/logs",
        json_format=True
    )
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test with extra data
    logger.info("Message with context", extra={
        'extra_data': {
            'user_id': 123,
            'action': 'fetch_price'
        }
    })
    
    # Test adapter
    context_logger = LoggerAdapter(logger, {"service_id": "svc-001"})
    context_logger.info("Message with service context")
    
    print("Logger test complete. Check /tmp/logs/test-service.log")
