"""Logging configuration for the application."""

import sys
from loguru import logger
from pathlib import Path

# Remove default logger
logger.remove()

# Add console logger with custom format
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

# Add file logger for production
log_file = Path("logs/app.log")
log_file.parent.mkdir(exist_ok=True)

logger.add(
    log_file,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    compression="zip"
)

# Add error file logger
error_log_file = Path("logs/error.log")
logger.add(
    error_log_file,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR",
    rotation="5 MB",
    retention="30 days",
    compression="zip"
)

# Export configured logger
__all__ = ["logger"]
