"""Application configuration.

All environment-specific values live here and are read from environment
variables (optionally via a local ``.env`` file).  Defaults are chosen so the
app runs out-of-the-box locally on SQLite; production overrides them with real
secrets and a Postgres URL, keeping local and server configuration identical
in shape.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # SQLAlchemy URL. Local default is a file-based SQLite DB; production sets
    # e.g. postgresql+psycopg://user:pass@db:5432/roughapp
    database_url: str = "sqlite:///data/roughapp.db"

    # JWT signing. MUST be overridden in production via SECRET_KEY env var.
    secret_key: str = "change-me-in-production"
    access_token_expire_days: int = 30
    guest_session_expire_hours: int = 24
    # How often the API process purges expired guest sessions (0 = startup only).
    guest_cleanup_interval_minutes: int = 60

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so the .env file is parsed only once."""
    return Settings()
