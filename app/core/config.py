from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "rag-with-evals"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    database_url: PostgresDsn
    db_pool_size: int = 10
    db_max_overflow: int = 10

    jwt_secret: str = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    storage_dir: Path = Path("data/uploads")
    max_upload_mb: int = Field(default=20, gt=0)
    max_pages: int = Field(default=1000, gt=0)

    chunk_max_tokens: int = Field(default=400, ge=50)
    chunk_overlap_tokens: int = Field(default=60, ge=0)
    tokenizer_encoding: str = "o200k_base"

    # "hashing" needs no API key and is for tests and offline development only:
    # its vectors capture shared words, not meaning.
    embedding_provider: Literal["openai", "hashing"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = Field(default=128, gt=0, le=2048)
    openai_api_key: SecretStr | None = None
    openai_timeout_seconds: float = Field(default=30.0, gt=0)

    redis_url: RedisDsn = RedisDsn("redis://localhost:6379/0")
    ingestion_max_retries: int = Field(default=4, ge=0)

    cors_origins: list[str] = []

    @field_validator("database_url")
    @classmethod
    def require_async_driver(cls, value: PostgresDsn) -> PostgresDsn:
        if value.scheme != "postgresql+asyncpg":
            raise ValueError("DATABASE_URL must use the postgresql+asyncpg:// scheme")
        return value

    @model_validator(mode="after")
    def check_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap_tokens >= self.chunk_max_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_MAX_TOKENS")
        # OpenAI caps the total tokens in one embeddings request.
        if self.embedding_batch_size * self.chunk_max_tokens > 300_000:
            raise ValueError(
                "EMBEDDING_BATCH_SIZE x CHUNK_MAX_TOKENS must not exceed 300,000 tokens"
            )
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
