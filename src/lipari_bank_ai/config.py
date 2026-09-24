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
    default_model: str = "big-pickle"  # "gpt-4o-mini"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768
    ollama_url: str = "http://localhost:11434"
    max_tokens_per_request: int = 2000
    jwt_secret: str


settings = Settings()  # raise at import if missing required