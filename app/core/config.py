"""Configuration management with Pydantic settings."""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:admin@db:5432/slackbot",
        description="Database connection URL"
    )
    db_echo: bool = Field(
        default=False,
        description="Enable SQL query logging"
    )
    
    # Redis Configuration
    redis_url: str = Field(
        default="redis://redis:6379/0",
        description="Redis connection URL"
    )
    
    # Slack Configuration
    slack_bot_token: str = Field(
        default="",
        alias="SLACK_VERIFICATION_TOKEN",
        description="Slack bot user OAuth token"
    )
    slack_signing_secret: str = Field(
        default="",
        alias="SLACK_SIGNING_SECRET",
        description="Slack app signing secret"
    )
    slack_client_id: str = Field(
        default="",
        alias="SLACK_CLIENT_ID",
        description="Slack app client ID"
    )
    slack_client_secret: str = Field(
        default="",
        alias="SLACK_CLIENT_SECRET",
        description="Slack app client secret"
    )
    
    # AI Configuration
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key"
    )
    
    # Application Configuration
    app_name: str = Field(
        default="slackbot-backend",
        description="Application name"
    )
    app_base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL for the application (used for OAuth redirects)"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from environment


# Global settings instance
settings = Settings()
