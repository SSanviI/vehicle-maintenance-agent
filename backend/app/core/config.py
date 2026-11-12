"""
Application configuration using Pydantic Settings.
Loads from environment variables and .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_env: str = Field(default="development", alias="APP_ENV")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    app_secret_key: str = Field(default="change-me-in-production", alias="APP_SECRET_KEY")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=7860, validation_alias="APP_PORT")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/vehicle_maintenance",
        alias="DATABASE_URL"
    )
    database_url_sync: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/vehicle_maintenance",
        alias="DATABASE_URL_SYNC"
    )

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # Primary LLM
    primary_llm_provider: str = Field(default="openai", alias="PRIMARY_LLM_PROVIDER")
    primary_llm_model: str = Field(default="gpt-4o-mini", alias="PRIMARY_LLM_MODEL")
    primary_llm_api_key: str = Field(default="", alias="PRIMARY_LLM_API_KEY")
    primary_llm_base_url: str = Field(default="https://api.openai.com/v1", alias="PRIMARY_LLM_BASE_URL")

    # Secondary LLM
    secondary_llm_provider: Optional[str] = Field(default=None, alias="SECONDARY_LLM_PROVIDER")
    secondary_llm_model: Optional[str] = Field(default=None, alias="SECONDARY_LLM_MODEL")
    secondary_llm_api_key: Optional[str] = Field(default=None, alias="SECONDARY_LLM_API_KEY")
    secondary_llm_base_url: Optional[str] = Field(default=None, alias="SECONDARY_LLM_BASE_URL")

    # Tertiary LLM
    tertiary_llm_provider: Optional[str] = Field(default=None, alias="TERTIARY_LLM_PROVIDER")
    tertiary_llm_model: Optional[str] = Field(default=None, alias="TERTIARY_LLM_MODEL")
    tertiary_llm_api_key: Optional[str] = Field(default=None, alias="TERTIARY_LLM_API_KEY")
    tertiary_llm_base_url: Optional[str] = Field(default=None, alias="TERTIARY_LLM_BASE_URL")

    # ML Model
    ml_model_path: str = Field(default="models/maintenance_model.joblib", alias="ML_MODEL_PATH")
    ml_pipeline_path: str = Field(default="models/preprocessing_pipeline.joblib", alias="ML_PIPELINE_PATH")

    # Google OAuth
    google_client_id: str = Field(default="", alias="GOOGLE_CLIENT_ID")
    google_client_secret: str = Field(default="", alias="GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str = Field(default="http://localhost:7860/auth/callback", alias="GOOGLE_REDIRECT_URI")
    frontend_url: str = Field(default="http://localhost:3000", alias="FRONTEND_URL")
    secure_cookies: bool = Field(default=False, alias="SECURE_COOKIES")

    # Context Management
    max_recent_messages: int = 20
    compaction_threshold: int = 30
    max_context_tokens: int = 4000

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.parent.parent

    def get_llm_configs(self) -> list:
        """Return list of configured LLM provider configs in priority order."""
        configs = []

        if self.primary_llm_api_key:
            configs.append({
                "name": "primary",
                "provider": self.primary_llm_provider,
                "model": self.primary_llm_model,
                "api_key": self.primary_llm_api_key,
                "base_url": self.primary_llm_base_url,
            })

        if self.secondary_llm_api_key:
            configs.append({
                "name": "secondary",
                "provider": self.secondary_llm_provider,
                "model": self.secondary_llm_model,
                "api_key": self.secondary_llm_api_key,
                "base_url": self.secondary_llm_base_url,
            })

        if self.tertiary_llm_api_key:
            configs.append({
                "name": "tertiary",
                "provider": self.tertiary_llm_provider,
                "model": self.tertiary_llm_model,
                "api_key": self.tertiary_llm_api_key,
                "base_url": self.tertiary_llm_base_url,
            })

        return configs


# Singleton
_settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
