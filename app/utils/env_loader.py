"""Environment variable loader that reads .env file on each request."""

import os
from pathlib import Path
from typing import Optional
from loguru import logger


def load_env_file() -> None:
    """Load environment variables from .env file."""
    env_file = Path("/app/.env")
    
    if not env_file.exists():
        logger.warning(".env file not found at /app/.env")
        return
    
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    os.environ[key] = value
                    
        logger.debug("Environment variables reloaded from .env file")
        
    except Exception as e:
        logger.error(f"Failed to load .env file: {str(e)}")


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable, reloading from .env file first."""
    load_env_file()
    return os.environ.get(key, default)


def get_env_var_required(key: str) -> str:
    """Get required environment variable, reloading from .env file first."""
    value = get_env_var(key)
    if value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    return value
