from decimal import Decimal
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "LipariBank AI"
    debug: bool = False

    database_url: str
    openai_api_key: str
    anthropic_api_key: str
    opencode_api_key: str
    default_model: str = "big-pickle"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768
    ollama_url: str = "http://localhost:11434"
    agent_model: str = "qwen2.5:7b"
    max_tokens_per_request: int = 2000
    jwt_secret: str
    soglia_approvazione_eur: Decimal = Decimal("5000")
    redis_url: str = ""


settings = Settings()  # raise at import if missing required