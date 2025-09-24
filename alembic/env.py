"""Alembic environment configuration."""

import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from alembic import context
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your models here to ensure they're registered with SQLAlchemy
try:
    # Import models but prevent auto-creation
    import os
    os.environ['SQLALCHEMY_SILENCE_UBER_WARNING'] = '1'
    
    from app.models.base import Base
    from app.core.config import settings
    
    # Override the database URL to use localhost for migrations
    settings.database_url = "postgresql://postgres:admin@localhost:5432/slackbot"
    
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    # Create a minimal Base if imports fail
    from sqlalchemy.ext.declarative import declarative_base
    Base = declarative_base()
    # Create minimal settings
    class Settings:
        database_url = "postgresql://postgres:admin@localhost:5432/slackbot"
    settings = Settings()

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_url():
    """Get database URL from settings."""
    # For local development, use localhost instead of 'db'
    url = settings.database_url
    if url.startswith('postgresql+asyncpg://'):
        # Convert async URL to sync URL and use localhost
        url = url.replace('postgresql+asyncpg://', 'postgresql://')
        url = url.replace('db:', 'localhost:')
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the script output.

    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with the given connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    # For now, we'll use synchronous engine for migrations
    # since Alembic doesn't fully support async yet
    from sqlalchemy import create_engine
    
    # Get the URL (already converted to sync and localhost)
    url = get_url()
    
    connectable = create_engine(
        url,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        do_run_migrations(connection)

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
