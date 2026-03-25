from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    chat_model: str = "local-rule-based"
    embedding_model: str = "local-lexical"
    data_dir: str = "./data"
    vectorstore_path: str = "./data/vectorstore"
    max_upload_mb: int = 8
    max_requirements: int = 8
    match_threshold: float = 0.28
    app_env: str = "local"
    log_json: bool = False
    metrics_window_size: int = 1000
    evaluation_cache_size: int = 256
    search_cache_size: int = 512

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
